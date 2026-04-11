# LM15 Completeness Report

- Required total: 16
- Required passed: 15
- Required failed: 0
- Required skipped: 1
- Score: 1.000

## Per provider

| provider | pass | fail | skip |
|---|---:|---:|---:|
| anthropic | 4 | 0 | 0 |
| gemini | 4 | 0 | 0 |
| openai | 4 | 0 | 0 |
| shared | 3 | 0 | 1 |

## Tests

| id | provider | probe | required | status | details |
|---|---|---|---:|---|---|
| openai.complete.fixture | openai | fixture_complete | true | pass | matched frozen fixture: openai.basic_text |
| anthropic.complete.fixture | anthropic | fixture_complete | true | pass | matched frozen fixture: anthropic.basic_text |
| gemini.complete.fixture | gemini | fixture_complete | true | pass | matched frozen fixture: gemini.basic_text |
| openai.stream.fixture | openai | fixture_stream | true | pass | matched frozen fixture: openai.basic_text_stream |
| anthropic.stream.fixture | anthropic | fixture_stream | true | pass | matched frozen fixture: anthropic.basic_text_stream |
| gemini.stream.fixture | gemini | fixture_stream | true | pass | matched frozen fixture: gemini.basic_text_stream |
| openai.tool.fixture | openai | fixture_tool_call | true | pass | matched frozen fixture: openai.basic_tool_call |
| openai.extended.fixture | openai | openai_extended | true | pass | openai extended endpoints normalized |
| anthropic.tool.fixture | anthropic | fixture_tool_call | true | pass | matched frozen fixture: anthropic.basic_tool_call |
| anthropic.extended.fixture | anthropic | anthropic_extended | true | pass | anthropic files+batches normalized |
| gemini.tool.fixture | gemini | fixture_tool_call | true | pass | matched frozen fixture: gemini.basic_tool_call |
| gemini.extended.fixture | gemini | gemini_extended | true | pass | gemini extended endpoints normalized |
| errors.mapping.fixture | shared | error_mapping | true | pass | error mapping matches frozen fixtures |
| adapter.contract.fixture | shared | adapter_contract | true | pass | all adapters expose full plugin contract |
| live.contract.fixture | shared | live_contract | true | pass | live contract fixtures roundtrip through canonical serde |
| transport.pycurl_stream.fixture | shared | transport_streaming | true | skip | pycurl not installed |
