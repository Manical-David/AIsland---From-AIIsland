using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace ClassIsland.AISmartClass.Services;

/// <summary>
/// AIChatService HTTP 请求实现：非流式与流式（SSE）请求。
/// </summary>
public partial class AIChatService
{
    private async Task<string> SendRequestAsync(
        string system, string user, AiRequestSnapshot snapshot, CancellationToken ct)
    {
        var jsonBody = JsonSerializer.Serialize(BuildChatBody(system, user, snapshot, stream: false));
        var content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

        using var request = new HttpRequestMessage(HttpMethod.Post, snapshot.Endpoint) { Content = content };
        request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {snapshot.ApiKey}");

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(snapshot.TimeoutSeconds));

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, timeoutCts.Token).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            await ThrowApiResponseExceptionAsync(response, timeoutCts.Token).ConfigureAwait(false);
        }

        var responseJson = await response.Content.ReadAsStringAsync(timeoutCts.Token).ConfigureAwait(false);
        try
        {
            using var doc = JsonDocument.Parse(responseJson);
            if (TryExtractCompletionText(doc.RootElement, out var text))
                return text.Trim();

            throw new AIResponseFormatException(
                "API 响应中未找到可识别的文本内容（支持 OpenAI Chat Completions 和 MiniMax 响应格式）",
                TruncateResponseBody(responseJson));
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException or IndexOutOfRangeException)
        {
            throw new AIResponseFormatException(
                "API 返回内容不是可识别的 Chat Completions JSON 格式（支持 OpenAI 和 MiniMax）",
                TruncateResponseBody(responseJson),
                ex);
        }
    }

    private static bool TryExtractCompletionText(JsonElement root, out string text)
    {
        text = "";
        if (root.ValueKind == JsonValueKind.Object &&
            root.TryGetProperty("choices", out var choices) &&
            choices.ValueKind == JsonValueKind.Array && choices.GetArrayLength() > 0)
        {
            var choice = choices[0];
            if (choice.ValueKind == JsonValueKind.Object)
            {
                if (choice.TryGetProperty("message", out var message) &&
                    message.ValueKind == JsonValueKind.Object &&
                    TryGetText(message, "content", out text))
                    return true;

                // 兼容 OpenAI Completions 以及 MiniMax 的部分旧版/代理响应。
                if (TryGetText(choice, "text", out text))
                    return true;

                // MiniMax 旧版 chatcompletion_v2 或兼容代理可能将回复放在 messages[].text。
                if (choice.TryGetProperty("messages", out var messages) &&
                    messages.ValueKind == JsonValueKind.Array)
                {
                    var parts = new List<string>();
                    foreach (var item in messages.EnumerateArray())
                    {
                        if (item.ValueKind != JsonValueKind.Object)
                            continue;

                        if (item.TryGetProperty("sender_type", out var senderType) &&
                            senderType.ValueKind == JsonValueKind.String &&
                            !string.Equals(senderType.GetString(), "BOT", StringComparison.OrdinalIgnoreCase))
                            continue;

                        if (TryGetText(item, "text", out var part) || TryGetText(item, "content", out part))
                            parts.Add(part);
                    }

                    if (parts.Count > 0)
                    {
                        text = string.Concat(parts);
                        return true;
                    }
                }
            }
        }

        // 某些 MiniMax 兼容网关会直接返回 assistant 消息，而不包 choices。
        return root.ValueKind == JsonValueKind.Object && TryGetText(root, "content", out text);
    }

    private static bool TryGetText(JsonElement element, string propertyName, out string text)
    {
        text = "";
        if (element.ValueKind != JsonValueKind.Object ||
            !element.TryGetProperty(propertyName, out var value))
            return false;

        if (value.ValueKind == JsonValueKind.String)
        {
            text = value.GetString() ?? "";
            return true;
        }

        // 兼容返回文本内容块数组的网关；忽略非文本的多模态内容块。
        if (value.ValueKind == JsonValueKind.Array)
        {
            var parts = new List<string>();
            foreach (var part in value.EnumerateArray())
            {
                if (part.ValueKind == JsonValueKind.Object && TryGetText(part, "text", out var partText))
                    parts.Add(partText);
            }

            text = string.Concat(parts);
            return true;
        }

        return false;
    }

    /// <summary>
    /// 发送流式请求并逐 token 返回。兼容 OpenAI Chat Completions SSE 格式。
    /// </summary>
    private async IAsyncEnumerable<string> SendStreamRequestAsync(
        string system, string user, AiRequestSnapshot snapshot, [EnumeratorCancellation] CancellationToken ct)
    {
        var jsonBody = JsonSerializer.Serialize(BuildChatBody(system, user, snapshot, stream: true));
        var content = new StringContent(jsonBody, Encoding.UTF8, "application/json");

        using var request = new HttpRequestMessage(HttpMethod.Post, snapshot.Endpoint) { Content = content };
        request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {snapshot.ApiKey}");
        request.Headers.TryAddWithoutValidation("Accept", "text/event-stream");

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(snapshot.TimeoutSeconds));
        var requestToken = timeoutCts.Token;

        using var response = await _http.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            requestToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
        {
            await ThrowApiResponseExceptionAsync(response, requestToken).ConfigureAwait(false);
        }

        await using var stream = await response.Content.ReadAsStreamAsync(requestToken).ConfigureAwait(false);
        using var reader = new StreamReader(stream, Encoding.UTF8);

        var completed = false;
        var emittedContent = false;
        while (true)
        {
            var line = await reader.ReadLineAsync(requestToken).ConfigureAwait(false);
            if (line == null) break;
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;

            var data = line["data:".Length..].Trim();
            if (data == "[DONE]")
            {
                completed = true;
                break;
            }

            string? token = null;
            try
            {
                using var doc = JsonDocument.Parse(data);
                var root = doc.RootElement;
                if (root.TryGetProperty("choices", out var choices) &&
                    choices.GetArrayLength() > 0)
                {
                    var choice = choices[0];
                    if (choice.TryGetProperty("finish_reason", out var finishReason) &&
                        finishReason.ValueKind == JsonValueKind.String &&
                        !string.IsNullOrEmpty(finishReason.GetString()))
                    {
                        completed = true;
                    }

                    if (choice.TryGetProperty("delta", out var delta) &&
                        TryGetText(delta, "content", out var deltaText))
                    {
                        token = deltaText;
                    }
                    else if (!emittedContent && choice.TryGetProperty("message", out var message) &&
                             TryGetText(message, "content", out var messageText))
                    {
                        // MiniMax 可能在流式响应末尾额外发送完整 completion 对象。
                        token = messageText;
                    }
                    else if (!emittedContent && TryGetText(choice, "text", out var choiceText))
                    {
                        token = choiceText;
                    }
                }
            }
            catch (JsonException)
            {
                // SSE 数据行不是合法 JSON（如某些厂商的心跳），忽略。
                continue;
            }

            if (!string.IsNullOrEmpty(token))
            {
                emittedContent = true;
                yield return token;
            }
        }

        if (!completed)
            throw new IOException("AI 流式响应在完成标记前中断");
    }

    private static async Task ThrowApiResponseExceptionAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        string? responseBody = null;
        try
        {
            responseBody = TruncateResponseBody(
                await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
        }
        catch (Exception ex)
        {
            responseBody = $"[读取 API 错误响应体失败: {ex.GetType().Name}: {ex.Message}]";
        }

        var requestId = TryGetHeader(response.Headers, "x-request-id") ??
                        TryGetHeader(response.Headers, "request-id") ??
                        TryGetHeader(response.Headers, "x-trace-id") ??
                        TryGetHeader(response.Headers, "trace-id");
        throw new AIHttpResponseException(
            response.StatusCode,
            response.ReasonPhrase,
            requestId,
            responseBody);
    }

    private static string TruncateResponseBody(string responseBody)
    {
        const int maxResponseBodyLength = 16 * 1024;
        return responseBody.Length > maxResponseBodyLength
            ? responseBody[..maxResponseBodyLength] + "\n...[响应体已截断]"
            : responseBody;
    }

    private static string? TryGetHeader(HttpResponseHeaders headers, string name)
    {
        return headers.TryGetValues(name, out var values)
            ? string.Join(", ", values)
            : null;
    }

    /// <summary>
    /// 构造 Chat Completions 请求体。对 DeepSeek 系列（含阿里百炼渠道托管的 DeepSeek）额外注入 thinking=disabled。
    /// DeepSeek（deepseek-flash / deepseek-v4.1-flash 等）默认开启思考模式，会先把思维链写入
    /// reasoning_content，若 max_tokens 较小，content 可能被挤空，导致流式解析不到
    /// 任何正文而触发「AI 流式响应内容为空」降级。本插件只需要简短文本，禁用思考模式
    /// 可获得稳定、更快的 content 输出。
    /// </summary>
    private static Dictionary<string, object?> BuildChatBody(
        string system, string user, AiRequestSnapshot snapshot, bool stream)
    {
        var body = new Dictionary<string, object?>
        {
            ["model"] = snapshot.Model,
            ["temperature"] = snapshot.Temperature,
            ["max_tokens"] = snapshot.MaxTokens,
            ["stream"] = stream,
            ["messages"] = new object[]
            {
                new { role = "system", content = system },
                new { role = "user", content = user }
            }
        };

        if (IsDeepSeekProvider(snapshot))
            body["thinking"] = new { type = "disabled" };

        return body;
    }

    private static bool IsDeepSeekProvider(AiRequestSnapshot snapshot) =>
        snapshot.Endpoint.Contains("deepseek", StringComparison.OrdinalIgnoreCase) ||
        snapshot.Model.StartsWith("deepseek-", StringComparison.OrdinalIgnoreCase);
}
