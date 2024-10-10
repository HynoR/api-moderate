package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v2"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

type Config struct {
	OpenAIAPIKey        string `yaml:"openai_api_key"`
	ModerationAPIURL    string `yaml:"moderation_api_url"`
	TargetURL           string `yaml:"target_url"`
	Port                int    `yaml:"port"`
	WarningMsg          string `yaml:"warning_msg"`
	MinCharsModerate    int    `yaml:"min_chars_moderate"`    // 达到多少字符时进行审核,不达到则绕过审核
	FullContextModerate bool   `yaml:"full_context_moderate"` // 是否对完整上下文进行审核,如果启用，使用全部上下文消息，否则取用户最新一条消息
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ModerationResponse struct {
	Results []struct {
		Flagged bool `json:"flagged"`
	} `json:"results"`
}

type OpenAIStyleResponse struct {
	ID                string `json:"id"`
	Object            string `json:"object"`
	Created           int64  `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Choices           []struct {
		Index        int               `json:"index"`
		Delta        map[string]string `json:"delta"`
		Logprobs     interface{}       `json:"logprobs"`
		FinishReason string            `json:"finish_reason"`
	} `json:"choices"`
}

var config Config

func init() {
	// Initialize configuration and logger
	initializeConfig()
	initializeLogger()
}

func initializeConfig() {
	configFile, err := os.ReadFile("config.yaml")
	if err != nil {
		slog.Error("读取配置文件错误", "错误信息", err)
		os.Exit(1)
	}

	err = yaml.Unmarshal(configFile, &config)
	if err != nil {
		slog.Error("解析配置文件错误", "错误信息", err)
		os.Exit(1)
	}
}

func initializeLogger() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(logger)
}

func getUserContent(messages []Message) string {
	userContents := make([]string, 0, len(messages))
	for _, msg := range messages {
		if msg.Role == "user" {
			userContents = append(userContents, msg.Content)
		}
	}
	return strings.Join(userContents, " ")
}

func moderateContent(content string) (bool, error) {
	slog.Info("正在审核内容", "内容", content)
	jsonData, err := json.Marshal(map[string]string{"input": content})
	if err != nil {
		return false, err
	}

	req, err := http.NewRequest("POST", config.ModerationAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return false, err
	}
	setHeaders(req, map[string]string{
		"Content-Type":  "application/json",
		"Authorization": fmt.Sprintf("Bearer %s", config.OpenAIAPIKey),
	})

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("moderation API returned status code %d", resp.StatusCode)
	}

	var moderationResp ModerationResponse
	if err := json.NewDecoder(resp.Body).Decode(&moderationResp); err != nil {
		return false, err
	}

	flagged := len(moderationResp.Results) > 0 && moderationResp.Results[0].Flagged
	slog.Info("审核结果", "是否标记", flagged)
	return flagged, nil
}

func setHeaders(req *http.Request, headers map[string]string) {
	for key, value := range headers {
		req.Header.Set(key, value)
	}
}

func logFlaggedContent(content string) {
	slog.Warn("标记为不合规的内容", "内容", content)
	if err := appendToFile("log.txt", content+"\n"); err != nil {
		slog.Error("写入日志文件错误", "错误信息", err)
	}
}

func appendToFile(filename, content string) error {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = file.WriteString(content)
	return err
}

func generateOpenAIStyleResponse(warningMessage, model string) OpenAIStyleResponse {
	if model == "" {
		model = "gpt-4o-mini"
	}
	return OpenAIStyleResponse{
		ID:                "chatcmpl-" + uuid.New().String()[:24],
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             model,
		SystemFingerprint: "fp_" + uuid.New().String()[:12],
		Choices: []struct {
			Index        int               `json:"index"`
			Delta        map[string]string `json:"delta"`
			Logprobs     interface{}       `json:"logprobs"`
			FinishReason string            `json:"finish_reason"`
		}{
			{
				Index:        0,
				Delta:        map[string]string{"content": warningMessage},
				Logprobs:     nil,
				FinishReason: "stop",
			},
		},
	}
}

func handleFlaggedContent(c *gin.Context, isStream bool, model string) {
	response := generateOpenAIStyleResponse(config.WarningMsg, model)
	if isStream {
		c.Header("Content-Type", "text/event-stream")
		c.Stream(func(w io.Writer) bool {
			jsonResp, _ := json.Marshal(response)
			c.SSEvent("", string(jsonResp))
			c.SSEvent("", "[DONE]")
			return false
		})
	} else {
		c.JSON(http.StatusOK, response)
	}
}

func handleChatCompletions(c *gin.Context) {
	slog.Info("收到聊天完成请求")

	// 读取原始请求体
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		slog.Error("读取请求体错误", "错误信息", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Error reading request body"})
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewBuffer(body))

	var chatReq struct {
		Model    string    `json:"model"`
		Messages []Message `json:"messages"`
		Stream   bool      `json:"stream"`
	}
	if err := json.Unmarshal(body, &chatReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	var userContent string
	if config.FullContextModerate {
		userContent = getUserContent(chatReq.Messages)
	} else {
		if len(chatReq.Messages) > 0 {
			lastMessage := chatReq.Messages[len(chatReq.Messages)-1]
			if lastMessage.Role == "user" {
				userContent = lastMessage.Content
			}
		}
	}
	if len(userContent) >= config.MinCharsModerate {
		flagged, err := moderateContent(userContent)
		if err != nil {
			slog.Error("审核错误", "错误信息", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Moderation error"})
			return
		}
		if flagged {
			logFlaggedContent(userContent)
			handleFlaggedContent(c, chatReq.Stream, chatReq.Model)
			return
		}
	}

	slog.Info("正在转发请求到目标URL")
	proxyRequest(c, body)
}

func proxyRequest(c *gin.Context, body []byte) {
	proxyReq, err := http.NewRequest("POST", config.TargetURL, bytes.NewBuffer(body))
	if err != nil {
		slog.Error("创建代理请求错误", "错误信息", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error forwarding request"})
		return
	}

	copyHeaders(c.Request.Header, proxyReq.Header)
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		slog.Error("转发请求错误", "错误信息", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error forwarding request"})
		return
	}
	defer resp.Body.Close()

	copyHeaders(resp.Header, c.Writer.Header())
	c.Status(resp.StatusCode)
	io.Copy(c.Writer, resp.Body)
}

func copyHeaders(src, dest http.Header) {
	for name, values := range src {
		for _, value := range values {
			dest.Add(name, value)
		}
	}
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Service Running...")
	})
	r.POST("/v1/chat/completions", handleChatCompletions)
	if err := r.Run(fmt.Sprintf(":%d", config.Port)); err != nil {
		slog.Error("启动服务器失败", "错误信息", err)
	}
}
