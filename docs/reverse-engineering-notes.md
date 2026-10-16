# 逆向分析记录

本文件记录本项目所依赖的事实是怎么查出来的，以及踩过的坑。所有结论都是 2026-10-16 在
**公开安装包** + 实测请求上得到的，不含任何未公开数据、不涉及修改客户端。

> 想直接看结论：读 [`templates/how.html`](../templates/how.html)（渲染为 `/how`）。
> 这里保留的是推导过程、证据和边界条件，方便以后接口变了能快速定位。

## 1. 安装包

| 项 | 值 |
|---|---|
| 官网 PC 页 | <https://shurufa.doubao.com/pc>（SPA，下载地址由接口给出） |
| 下载信息接口 | `GET https://shurufa.doubao.com/api/v1/app/download_url?platform=windows` |
| Windows 包 | `DoubaoIME_Installer_0.9.2.7_release.exe`，123,961,912 字节 |
| SHA256 | `d777027cc822a67dac04ba502616f6e6216a54c06681e21f8f45ef0a89cec5db` |
| 版本 | V0.9.2（version_code 90207，发布日 2026-10-10） |
| 打包器 | **Inno Setup 6.7.0** |

`platform` 还支持 `macos` / `android` / `ios` / `harmony`（macOS 包 193 MB，内含 174 MB 的 `DoubaoIme.zip`；Android APK 171 MB）。

### 解包

`innoextract` 1.9（官方最新）与 master（支持到 6.4.0.1）都读不了 6.7.0 的包。可行做法是
**32 位 Wine + 虚拟显示跑静默安装**：

```bash
dpkg --add-architecture i386 && apt-get update
apt-get install -y --no-install-recommends wine32:i386 xvfb

export WINEPREFIX=/tmp/wineprefix WINEARCH=win32
xvfb-run -a /usr/lib/wine/wine DoubaoIME_Installer_0.9.2.7_release.exe \
    /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG='C:\inno.log'
# 文件在 $WINEPREFIX/drive_c/Program Files/DoubaoIME/versions/v0.9.2.7/
```

坑：

- 必须用 `/usr/lib/wine/wine`（32 位加载器）；`/usr/local/bin/wine` 是 wine64，跑不了 32 位 PE。
- 没有 X 显示会报 `Error reading FPreparingMemo.Lines.Strings: System Error. Code: 1400`，要用 `xvfb-run`。
- 安装过程最后会因 TSF 注册失败（`register_new_tsf ... helper_start_failed`）**回滚版本目录**，
  但文件此时已经落盘，直接拷走即可。

## 2. 包结构与"哪部分负责语音识别"

`Program Files\DoubaoIME\versions\v0.9.2.7\`（约 267 MB）：

| 文件 | 大小 | 职责 |
|---|---|---|
| `ImeService.exe` | 25 MB | 服务进程：音频采集、驱动语音会话（`SamiAsrClientWin`）、AI 联想、上下文构造、日志上报 |
| `audioeffect-mt.dll` | 2.9 MB | **语音识别核心：字节 SAMI 语音 SDK**（Opus、WebSocket/Cronet、protobuf、frontier 连接池、token、在线/整段两种模式调度） |
| `tsf-oime.dll` / `tsf-oime-core.dll` | 2.2 MB | TSF 文本服务（输入法本体、语音上屏、选中文字语音改写 `voice_modify_*`） |
| `ui.dll` | 5 MB | 输入法界面 |
| `onnxruntime.dll` | 10 MB | 本地 ONNX 推理（手写 CRNN、候选打分 MLP，**不参与语音识别**） |
| `sscronet.dll` | 9.9 MB | Cronet/TTNet 网络栈 |
| `ime_net_sdk.dll` | 7 MB | Rust 网络 SDK（QUIC、DNS 调度），内含客户端证书 `CN=ImeNetSDK` |
| `rpc.dll` / `doubaoime-rpc-new.dll` | 3.3 MB | 进程间 RPC |
| `applogrs.dll` / `parfait*` | — | 埋点 / 崩溃上报 |
| `files/data/dict/*` | ~170 MB | 词库与模型，文件名是**内容哈希**；`asr/` 下是识别的文本后处理词表（n-gram/纠错/豁免），**不是声学模型** |

二进制里残留的编译期路径给出了源码结构（等于白送目录树）：

- SAMI：`C:\19626\sami_core\sami_interface\src\capi\speech_v2\asr\online\{asr_on_worker,asr_on_callback,protocol,send_queue,service_wrapper}`
  + `common_util\net\{frontier,connect_pool_v2,nqe,transport}` + `common_util\protobuf\WebSocket{Request,Response}.pb.cc`
- 输入法本体：`C:\Workfolder\oime\obric-ime5\src\{server,shell,ime_ui,common,ttnet}\*`
  （`sami_asr_client_win.cpp`、`asr_context_builder.cpp`、`voice_key_hook.cpp`、`asr_modify_pair_uploader.cpp`）

## 3. 凭据链路（为什么不用登录）

```
① POST https://log.snssdk.com/service/2/device_register/
     body: {magic_tag:"ss_app_log", header:{aid:685343, app_name:"oime", cdid/openudid/clientudid, …}}
     resp: {device_id, install_id, device_token}
② POST https://is.snssdk.com/service/settings/v3/?device_id=&cid=&cdid=…
     header: x-ss-stub = md5("body=null").upper()
     resp: data.settings.asr_config.app_key

aid（编译进客户端，应用级） + app_key（按设备下发） → 服务端据此计量，与用户账号无关
```

- `aid = 685343`（取自 `ImeService.exe` 字符串池）
- UA 需伪装成官方客户端：`com.bytedance.android.doubaoime/100102018 (Linux; U; Android 16; en_US; Pixel 7 Pro; Build/BP2A.250605.031.A2)`
- `app_key` 会随设备/时间变化，建议缓存 12 小时；**不要下发到前端**

## 4. 识别协议

一条 WebSocket，二进制 protobuf：

```
wss://frontier-audio-ime-ws.doubao.com/ocean/api/v1/ws?aid=685343&device_id=<did>
headers: User-Agent: …doubaoime…   proto-version: v2   x-custom-keepalive: true

① StartTask     {token=app_key, service=ASR}      → TaskStarted
② StartSession  {payload=会话参数 JSON}            → SessionStarted
③ TaskRequest×N {audio=20ms PCM, frame_state}      → 结果 JSON（累积文本）
④ FinishSession                                    → SessionFinished
```

请求字段（实测）：`2=token`、`3=service_name("ASR")`、`5=method_name`、`6=payload(JSON 字符串)`、
`7=audio_data`、`8=request_id`、`9=frame_state`（1 首帧 / 3 中间 / 9 末帧）。
响应字段：`2=task_id`、`4=event`、`5=status_code`、`6=status_text`、`7=result JSON`。

`StartSession` 的 payload 关键项：

```json
{"audio_info":{"channel":1,"format":"raw","sample_rate":16000},
 "enable_punctuation":true,"enable_speech_rejection":false,
 "extra":{"did":"…","input_mode":"tool","strong_ddc":true,"use_twopass_retry":true,
          "enable_asr_twopass":true,"enable_asr_threepass":true,
          "disable_user_words":false,"cell_compress_rate":8,
          "context":"<base64: chat + ime_info + hotwords>"}}
```

结果 JSON：`results[]{text, start_time, end_time, confidence, is_interim, is_vad_finished, index, alternatives[].words[]}`。
`text` 是**这一段发言的累积文本**，最后一条即完整结果。

### 协议相关的坑

1. **没有 `proto-version: v2` 头时，上游回 JSON 而不是 protobuf**（前端层做了兼容）。
   而浏览器无法设置 WebSocket 自定义头 → **网页端不可能直连，必须服务端中转**。
2. `50700000 service discovery failure` 是**随机抖动**：同一 payload 实测约 1/5 概率失败，
   直连上游也一样（曾误判为 payload 问题，做了 A/B 才排除）。重试整段即可。
3. 其它状态码：`40200011` 并发配额满（重试无用）、`40200001` 鉴权失败（`IllegalToken`）、
   `40100003` `StartTaskExpected`（顺序错）、`40000022` `IllegalPayload`（payload 必须是字符串）。
4. 上游偶发把 `start_time` 回成 `-0.001`（表示未知），直接用会显示 `-0.0`；要用已知音频长度封顶。

## 5. 文本切句与时间戳（本项目里的坑）

1. **短语气词会被反复改字**：实测说一个「哎」，上游连续回 `唉 → 哎 → 唉` 三条，
   **`start_time` 完全相同（1.33s），只有 `end_time` 在涨**。旧逻辑只看"文本有没有变长"，
   会把改字误判成新的一段，一句话变三行。
   → 判据改成：新结果起点与当前段起点几乎相同（≤0.35s）一律视为同段改字。
2. **合并要小心**：相邻（≤0.6s）且相近或极短的行可以合并（修上面的问题），
   但**必须跨上游消息**（用消息序号区分），否则"连说三遍同一句话"会被并成一句。
3. `alternatives[].words[].end_time` 偏大（带前瞻窗口）：6.28s 的音频里末字能回 12.56s。
   → 句尾优先取"下一句的首字时间"，末句用整段 `end_time`，再用已发送音频长度封顶。
4. `words` **不含标点**（text 63 字 / words 54 字，差的 9 个正是标点），对齐要按非标点字符顺序映射。
5. 停顿较久后上游会把累积文本**清零重来**（`start_time` 跳到上一段之后），
   要据此把上一段"锁住"，否则前面的内容会被覆盖。

## 6. 语言能力

模型是 **zh + en 双语**（官方宣传的中英混输），协议里没有 `language` 字段，
客户端创建句柄时写死的是 `speech_opus` + `zh`。

| 输入 | 输出 |
|---|---|
| 普通话 | 正确 |
| 英语（JFK 演讲） | 正确，标点大小写都对 |
| 中文 + 英文混说 | 各自正确 |
| 日语「こんにちは」 | 「国内计划。」 |
| 日语「私は日本語を勉強しています」 | 「马达西」+「我倒希望你哄我。」（声学层听到了，语言模型只有中文） |

## 7. 豆包的端侧离线识别（拿不到，但值得知道）

Windows 0.9.2.7 里**没有**这套东西；Android 1.5.1 与 macOS 1.0.2 有。

| 层 | 证据 | Win | mac | Android |
|---|---|---|---|---|
| 端侧推理引擎 **Flute** | `flute/engine/{engine,executor,flute_c_api}.cc`、`flute/framework/*`、`flute/base/filesystem_mmap/filesystem_packed_mmap.cc`（打包 mmap 模型容器） | ❌ | ✅ 主程序内 | ✅ `libaudioeffect.so` |
| ASR 模型结构 | `flute/module/asr/ntp_cif/{encoder_actor,cif_calculator,beam_searcher,decoder_context/prompt_processor}`、`feature/{mel_bank,panther,subband}_feature_extractor`、`module/vad/v4`、`module/signal/aec` | ❌ | ✅ | ✅ |
| SAMI 本地 worker | `speech_v2/asr/offline/{asr_off_singleton,asr_off_worker}.cpp`、`capi/flute/asr/flute_asr.cpp` | ❌ 只有 `online/*` | ✅ | ✅ |
| 模型下载管理 | 应用侧 `OimeOfflineModel`：`ASROfflineModel{Manager,FileStore,PathResolver,MetaInfoService,Manifest}`、通知 `start/pause/resume/retry/requestDelete`、日志 `ASR offline model path:`、缺模型弹窗 | ❌ | ✅ | — |

模型是 **PANTHER Conformer 编码器 + LLM 风格解码器**、CIF 机制、int8 量化，
逐张量 `.bin` + `*_shape.dat` + `config.json`/`token_config.json`，由 packed mmap 容器加载，
CPU int8（`PackBS8: 4x4 tiled layout…`）+ GPU kernel（macOS 用 Metal）。

在线/离线的融合策略（SAMI 内）：`PutOnlineCacheAudioToOffline`（把在线缓存的音频交给整段识别重跑）、
`[online vs offline] words num`（两份结果对比择优）、`asr has no network, switch to offline`、
`finish_wait_offline_time` / `optimize switch` 等时间参数。

**拿不过来的原因**：模型私有逐张量格式（引擎里写死张量名，版本一变就对不上）、
Flute 静态编译进客户端且无公开 API、模型包要鉴权下载。要离线就用开源等价物。

## 8. 本项目踩过的其它坑

- **路由被批量替换误删**：用 `s[s.index('@app.get("/api/lexicon")'):]` 做替换时取到了文件末尾，
  把 `/`、`/how`、`/favicon.ico` 一起删了，表现为整站 404。
- **localStorage 里存的行没带 `type` 字段**，渲染时被 `e.type === 'line'` 过滤掉，页面空白。
- **`log.info` 默认被 uvicorn 的日志配置吞掉**，排查时看不到，需显式 `basicConfig`。
- 本机 pip 受 PEP 668 限制，项目内自建 venv。
- 服务绑 `127.0.0.1` 时容器里的反代连不上，绑 docker 网桥地址（如 `172.17.0.1`）才通。
