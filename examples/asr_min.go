// 豆包输入法语音识别 · 最小可用客户端（Go）
//
// 用法:
//   cd examples && go mod tidy && go run asr_min.go <16k-mono-16bit.wav>
//
// 依赖: github.com/gorilla/websocket
package main

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

const (
	aid          = 685343 // 豆包输入法的 app id（取自客户端 ImeService.exe）
	appName      = "oime"
	versionCode  = 100102018
	versionName  = "1.1.2"
	userAgent    = "com.bytedance.android.doubaoime/100102018 (Linux; U; Android 16; en_US; Pixel 7 Pro; Build/BP2A.250605.031.A2)"
	registerURL  = "https://log.snssdk.com/service/2/device_register/"
	settingsURL  = "https://is.snssdk.com/service/settings/v3/"
	wsURL        = "wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws"
	sampleRate   = 16000
	frameBytes   = 640 // 20ms × 16kHz × 2 字节
	maxRetry     = 4
)

var httpClient = &http.Client{Timeout: 25 * time.Second}

// ---------------------------------------------------------------- 1. 拿凭据

func postJSON(endpoint string, params url.Values, body []byte, headers map[string]string, out any) error {
	req, err := http.NewRequest(http.MethodPost, endpoint+"?"+params.Encode(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s HTTP %s: %s", endpoint, resp.Status, string(raw))
	}
	return json.Unmarshal(raw, out)
}

// getCredentials 匿名注册一台设备并换取 ASR 的 app_key（不需要账号登录）
func getCredentials() (deviceID, appKey string, err error) {
	cdid := uuid.NewString()
	openudid := strings.ReplaceAll(uuid.NewString(), "-", "")[:16]
	clientudid := uuid.NewString()

	header := map[string]any{
		"device_id": 0, "install_id": 0, "aid": aid, "app_name": appName,
		"version_code": versionCode, "version_name": versionName,
		"manifest_version_code": versionCode, "update_version_code": versionCode,
		"channel": "official", "device_platform": "android", "os": "android", "os_api": "34",
		"os_version": "16", "device_type": "Pixel 7 Pro", "device_brand": "google",
		"device_model": "Pixel 7 Pro", "resolution": "1080*2400", "dpi": "420",
		"language": "zh", "timezone": 8, "access": "wifi", "region": "CN",
		"tz_name": "Asia/Shanghai", "tz_offset": 28800, "openudid": openudid,
		"clientudid": clientudid, "cdid": cdid,
	}
	body, _ := json.Marshal(map[string]any{
		"magic_tag": "ss_app_log", "header": header, "_gen_time": time.Now().UnixMilli(),
	})
	params := url.Values{
		"device_platform": {"android"}, "os": {"android"}, "ssmix": {"a"},
		"_rticket": {fmt.Sprint(time.Now().UnixMilli())}, "cdid": {cdid}, "channel": {"official"},
		"aid": {fmt.Sprint(aid)}, "app_name": {appName}, "version_code": {fmt.Sprint(versionCode)},
		"version_name": {versionName}, "manifest_version_code": {fmt.Sprint(versionCode)},
		"update_version_code": {fmt.Sprint(versionCode)}, "resolution": {"1080*2400"}, "dpi": {"420"},
		"device_type": {"Pixel 7 Pro"}, "device_brand": {"google"}, "language": {"zh"},
		"os_api": {"34"}, "os_version": {"16"}, "ac": {"wifi"},
	}
	var reg struct {
		DeviceIDStr string `json:"device_id_str"`
		DeviceID    any    `json:"device_id"`
	}
	if err = postJSON(registerURL, params, body,
		map[string]string{"Content-Type": "application/json", "User-Agent": userAgent}, &reg); err != nil {
		return "", "", err
	}
	deviceID = reg.DeviceIDStr
	if deviceID == "" || deviceID == "0" {
		deviceID = fmt.Sprint(reg.DeviceID)
	}
	if deviceID == "" || deviceID == "0" {
		return "", "", errors.New("设备注册失败：没有 device_id")
	}

	stub := []byte("body=null")
	sum := md5.Sum(stub)
	params = url.Values{
		"device_platform": {"android"}, "os": {"android"}, "ssmix": {"a"}, "channel": {"official"},
		"aid": {fmt.Sprint(aid)}, "app_name": {appName}, "version_code": {fmt.Sprint(versionCode)},
		"version_name": {versionName}, "device_id": {deviceID}, "cdid": {cdid},
		"_rticket": {fmt.Sprint(time.Now().UnixMilli())},
	}
	var settings struct {
		Data struct {
			Settings struct {
				ASRConfig struct {
					AppKey string `json:"app_key"`
				} `json:"asr_config"`
			} `json:"settings"`
		} `json:"data"`
	}
	if err = postJSON(settingsURL, params, stub, map[string]string{
		"User-Agent":       userAgent,
		"x-ss-stub":        strings.ToUpper(hex.EncodeToString(sum[:])),
		"Content-Type":     "application/x-www-form-urlencoded",
	}, &settings); err != nil {
		return "", "", err
	}
	appKey = settings.Data.Settings.ASRConfig.AppKey
	if appKey == "" {
		return "", "", errors.New("settings 没给 asr_config.app_key")
	}
	return deviceID, appKey, nil
}

// ---------------------------------------------------------------- 2. protobuf 帧

func putVarint(buf *bytes.Buffer, value uint64) {
	for {
		b := byte(value & 0x7f)
		value >>= 7
		if value != 0 {
			buf.WriteByte(b | 0x80)
		} else {
			buf.WriteByte(b)
			return
		}
	}
}

func putBytes(buf *bytes.Buffer, field int, raw []byte) {
	if len(raw) == 0 {
		return
	}
	putVarint(buf, uint64(field<<3|2))
	putVarint(buf, uint64(len(raw)))
	buf.Write(raw)
}

// encode 上游请求帧：2=token 3=service 5=method 6=payload 7=audio 8=request_id 9=frame_state
func encode(token, method, payload string, audio []byte, requestID string, state int) []byte {
	var buf bytes.Buffer
	putBytes(&buf, 2, []byte(token))
	putBytes(&buf, 3, []byte("ASR"))
	putBytes(&buf, 5, []byte(method))
	putBytes(&buf, 6, []byte(payload))
	putBytes(&buf, 7, audio)
	putBytes(&buf, 8, []byte(requestID))
	if state != 0 {
		putVarint(&buf, uint64(9<<3))
		putVarint(&buf, uint64(state))
	}
	return buf.Bytes()
}

func readVarint(data []byte, pos int) (uint64, int) {
	var value uint64
	var shift uint
	for {
		b := data[pos]
		pos++
		value |= uint64(b&0x7f) << shift
		shift += 7
		if b&0x80 == 0 {
			return value, pos
		}
	}
}

// decode 响应帧 → 字段号到值的映射（2=task_id 4=事件 5=状态码 6=状态文本 7=结果 JSON）
func decode(data []byte) map[int][][]byte {
	out := map[int][][]byte{}
	for off := 0; off < len(data); {
		tag, next := readVarint(data, off)
		off = next
		field, wire := int(tag>>3), int(tag&7)
		switch wire {
		case 0:
			val, next := readVarint(data, off)
			off = next
			raw := make([]byte, 8)
			binary.LittleEndian.PutUint64(raw, val)
			out[field] = append(out[field], raw)
		case 2:
			length, next := readVarint(data, off)
			off = next
			out[field] = append(out[field], data[off:off+int(length)])
			off += int(length)
		case 1:
			off += 8
		case 5:
			off += 4
		default:
			return out
		}
	}
	return out
}

func textOf(msg map[int][][]byte, field int) string {
	values := msg[field]
	if len(values) == 0 {
		return ""
	}
	return string(values[0])
}

func codeOf(msg map[int][][]byte) uint64 {
	values := msg[5]
	if len(values) == 0 {
		return 0
	}
	return binary.LittleEndian.Uint64(values[0])
}

// ---------------------------------------------------------------- 3. 跑一次识别

func readWav(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 44 || string(raw[0:4]) != "RIFF" {
		return nil, errors.New("不是 WAV 文件")
	}
	for off := 12; off+8 <= len(raw); {
		id := string(raw[off : off+4])
		size := int(binary.LittleEndian.Uint32(raw[off+4 : off+8]))
		if id == "fmt " {
			channels := binary.LittleEndian.Uint16(raw[off+10 : off+12])
			rate := binary.LittleEndian.Uint32(raw[off+12 : off+16])
			bits := binary.LittleEndian.Uint16(raw[off+22 : off+24])
			if channels != 1 || rate != sampleRate || bits != 16 {
				return nil, fmt.Errorf("需要 16kHz/单声道/16bit 的 WAV，实际 %dHz/%dch/%dbit", rate, channels, bits)
			}
		}
		if id == "data" {
			end := off + 8 + size
			if end > len(raw) {
				end = len(raw)
			}
			return raw[off+8 : end], nil
		}
		off += 8 + size + size%2
	}
	return nil, errors.New("WAV 里没有 data 段")
}

type result struct {
	Results []struct {
		Text string `json:"text"`
	} `json:"results"`
}

func transcribeOnce(path string) (string, error) {
	pcm, err := readWav(path)
	if err != nil {
		return "", err
	}
	deviceID, appKey, err := getCredentials()
	if err != nil {
		return "", err
	}
	requestID := uuid.NewString()

	header := http.Header{}
	header.Set("User-Agent", userAgent)
	header.Set("proto-version", "v2")
	header.Set("x-custom-keepalive", "true")
	target := fmt.Sprintf("%s?aid=%d&device_id=%s", wsURL, aid, deviceID)
	conn, _, err := websocket.DefaultDialer.Dial(target, header)
	if err != nil {
		return "", err
	}
	defer conn.Close()

	// ① StartTask
	if err = conn.WriteMessage(websocket.BinaryMessage, encode(appKey, "StartTask", "", nil, requestID, 0)); err != nil {
		return "", err
	}
	_, data, err := conn.ReadMessage()
	if err != nil {
		return "", err
	}
	if msg := decode(data); textOf(msg, 4) != "TaskStarted" {
		return "", fmt.Errorf("StartTask 失败: %s %d %s", textOf(msg, 4), codeOf(msg), textOf(msg, 6))
	}

	// ② StartSession
	sessionPayload, _ := json.Marshal(map[string]any{
		"audio_info":              map[string]any{"channel": 1, "format": "raw", "sample_rate": sampleRate},
		"enable_punctuation":      true,
		"enable_speech_rejection": false,
		"extra": map[string]any{
			"did": deviceID, "input_mode": "tool",
			"strong_ddc": true, "use_twopass_retry": true,
			"enable_asr_twopass": true, "enable_asr_threepass": true,
			"disable_user_words": false, "cell_compress_rate": 8,
		},
	})
	if err = conn.WriteMessage(websocket.BinaryMessage,
		encode(appKey, "StartSession", string(sessionPayload), nil, requestID, 0)); err != nil {
		return "", err
	}
	if _, data, err = conn.ReadMessage(); err != nil {
		return "", err
	}
	if msg := decode(data); textOf(msg, 4) != "SessionStarted" {
		return "", fmt.Errorf("StartSession 失败: %s %d %s", textOf(msg, 4), codeOf(msg), textOf(msg, 6))
	}

	// ③ 后台收结果，主线程按 20ms 一帧送音频
	text := make(chan string, 1)
	done := make(chan error, 1)
	go func() {
		var last string
		for {
			_, data, err := conn.ReadMessage()
			if err != nil {
				done <- err
				return
			}
			msg := decode(data)
			if raw := textOf(msg, 7); raw != "" {
				var parsed result
				if json.Unmarshal([]byte(raw), &parsed) == nil {
					for _, item := range parsed.Results {
						if item.Text != "" {
							last = item.Text // 累积文本，最后一条即完整结果
						}
					}
				}
			}
			switch textOf(msg, 4) {
			case "SessionFinished", "TaskFinished":
				text <- last
				return
			case "SessionFailed", "TaskFailed":
				done <- fmt.Errorf("识别失败: %d %s", codeOf(msg), textOf(msg, 6))
				return
			}
		}
	}()

	frames := (len(pcm) + frameBytes - 1) / frameBytes
	for i := 0; i < frames; i++ {
		start := i * frameBytes
		end := start + frameBytes
		if end > len(pcm) {
			end = len(pcm)
		}
		state := 3
		if i == 0 {
			state = 1
		} else if i == frames-1 {
			state = 9
		}
		if err = conn.WriteMessage(websocket.BinaryMessage,
			encode("", "TaskRequest", "{}", pcm[start:end], requestID, state)); err != nil {
			return "", err
		}
		time.Sleep(20 * time.Millisecond) // 按实时速度喂
	}

	// ④ FinishSession
	if err = conn.WriteMessage(websocket.BinaryMessage,
		encode(appKey, "FinishSession", "", nil, requestID, 0)); err != nil {
		return "", err
	}
	select {
	case out := <-text:
		return out, nil
	case err := <-done:
		return "", err
	case <-time.After(25 * time.Second):
		return "", errors.New("等待结果超时")
	}
}

func retryable(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "service discovery") || strings.Contains(msg, "失败: 5")
}

func transcribe(path string) (string, error) {
	var lastErr error
	for attempt := 0; attempt < maxRetry; attempt++ {
		out, err := transcribeOnce(path)
		if err == nil {
			return out, nil
		}
		lastErr = err
		if !retryable(err) {
			return "", err
		}
		fmt.Fprintf(os.Stderr, "  [重试 %d/%d] %v\n", attempt+1, maxRetry-1, err)
		time.Sleep(600 * time.Millisecond)
	}
	return "", lastErr
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "用法: go run asr_min.go <16k-mono-16bit.wav>")
		os.Exit(1)
	}
	out, err := transcribe(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "失败:", err)
		os.Exit(2)
	}
	fmt.Println(out)
}
