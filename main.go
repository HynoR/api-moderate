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
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

var httpClient = &http.Client{
	Timeout: 180 * time.Second, // 设定超时时间，防止请求挂起
}

var fileMutex sync.RWMutex

type Config struct {
	OpenAIAPIKey        string   `yaml:"openai_api_key"`
	ModerationAPIURL    string   `yaml:"moderation_api_url"`
	TargetURL           string   `yaml:"target_url"`
	Port                int      `yaml:"port"`
	WarningMsg          string   `yaml:"warning_msg"`
	MinCharsModerate    int      `yaml:"min_chars_moderate"`    // 达到多少字符时进行审核,不达到则绕过审核
	FullContextModerate bool     `yaml:"full_context_moderate"` // 是否对完整上下文进行审核,如果启用，使用全部上下文消息，否则取用户最新一条消息
	WhiteListModels     []string `yaml:"white_list_models"`     // 白名单模型,绕过审核
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
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint"`
	Choices           []Choice `json:"choices"`
}

type Choice struct {
	Index        int               `json:"index"`
	Delta        map[string]string `json:"delta"`
	Logprobs     interface{}       `json:"logprobs"`
	FinishReason string            `json:"finish_reason"`
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
		if msg.Role == "user" || msg.Role == "system" {
			userContents = append(userContents, msg.Content)
		}
	}
	return strings.Join(userContents, " ")
}

func moderateContent(content string) (bool, error) {
	model := "text-moderation-latest"
	if len(content) < 4096 {
		model = "omni-moderation-latest"
	}
	jsonData, err := json.Marshal(map[string]string{
		"model": model,
		"input": content})
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

	resp, err := httpClient.Do(req)
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
	slog.Info("审核结果", "是否标记", flagged, "模型", model)
	if flagged {
		slog.Info("审核拦截", "内容", content)
	}
	return flagged, nil
}

func setHeaders(req *http.Request, headers map[string]string) {
	for key, value := range headers {
		req.Header.Set(key, value)
	}
}

func logFlaggedContent(content string) {
	slog.Warn("标记为不合规的内容", "内容", content)
	fileMutex.Lock()
	defer fileMutex.Unlock()
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
		Choices: []Choice{
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

type OpenAIChatReq struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
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

	var chatReq OpenAIChatReq
	if err := json.Unmarshal(body, &chatReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	userContent := getUserContent(chatReq.Messages)
	// 检查长度是否超过8192字符，如果超过替换model值
	var replaceModel string
	if len(userContent) > 10*1024 && len(userContent) < 100*1024 {
		replaceModel = "glm-4-air"
	}
	if len(userContent) > 100*1024 {
		replaceModel = "glm-4-flash"
	}
	if replaceModel != "" {
		slog.Warn("请求体超过设定字符，替换model值", "新model", replaceModel, "字符数", len(userContent))
		newBody, err := replaceModelValue(body, replaceModel)
		if err != nil {
			slog.Error("替换model值错误", "错误信息", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Error replacing model value"})
			return
		}
		body = newBody
	}

	// 检查模型是否在白名单中
	for _, model := range config.WhiteListModels {
		if model == chatReq.Model {
			slog.Info("模型在白名单中，绕过审核", "模型", chatReq.Model)
			proxyRequest(c, body)
			return
		}
	}

	if !config.FullContextModerate {
		if len(chatReq.Messages) > 0 {
			for i := len(chatReq.Messages) - 1; i >= 0; i-- {
				if chatReq.Messages[i].Role == "user" {
					userContent = chatReq.Messages[i].Content
					break
				}
			}
			//lastMessage := chatReq.Messages[len(chatReq.Messages)-1]
			//if lastMessage.Role == "user" {
			//	userContent = lastMessage.Content
			//}
		}
	}
	if len(userContent) >= config.MinCharsModerate {
		userContents := splitText(userContent)
		if len(userContents) > 1 {
			slog.Info("内容过长，已分割", "分割片数", len(userContents))
		}
		for _, userContent := range userContents {
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
	}

	proxyRequest(c, body)
}

// 以48K字符切割一个文本
func splitText(text string) []string {
	if len(text) <= 48000 {
		return []string{text}
	}
	var result []string
	for i := 0; i < len(text); i += 48000 {
		end := i + 48000
		if end > len(text) {
			end = len(text)
		}
		result = append(result, text[i:end])
	}
	return result
}

// replaceModelValue 用于替换 JSON 数据中的 model 值 定制功能
func replaceModelValue(input []byte, newModelValue string) ([]byte, error) {
	// 使用通用的 map 来解析 JSON
	var data map[string]interface{}
	if err := json.Unmarshal(input, &data); err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}

	// 检查并替换 model 值
	if model, exists := data["model"]; exists {
		if _, ok := model.(string); ok {
			data["model"] = newModelValue
		}
	}

	output, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("could not encode JSON: %w", err)
	}

	return output, nil
}

func proxyRequest(c *gin.Context, body []byte) {
	slog.Info("正在转发请求到目标URL")

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
		dest[name] = values
	}
}

// 读取log.txt 解析为网页内容
func GetBanndedContent(c *gin.Context) {
	fileMutex.RLock()
	defer fileMutex.RUnlock()
	file, err := os.ReadFile("log.txt")
	if err != nil {
		c.JSON(http.StatusOK, gin.H{"error": "读取文件错误"})
		return
	}
	// 将文件内容转换为字符串，并将 '\n' 替换为 '<br>'
	htmlContent := strings.ReplaceAll(string(file), "\n", "<br>")

	// 将内容嵌入 HTML 中，确保换行符正常显示
	htmlResponse := fmt.Sprintf("<html><body>%s</body></html>", htmlContent)

	// 返回带有日志内容的 HTML
	c.Data(http.StatusOK, "text/html; charset=utf-8", []byte(htmlResponse))
}

func main() {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Service Running...")
	})
	r.POST("/v1/chat/completions", handleChatCompletions)
	r.GET("/api/getBannedContent", GetBanndedContent)
	if err := r.Run(fmt.Sprintf(":%d", config.Port)); err != nil {
		slog.Error("启动服务器失败", "错误信息", err)
	}
}
