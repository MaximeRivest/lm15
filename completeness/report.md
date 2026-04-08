# LM15 Completeness Report

- Required total: 15
- Required passed: 14
- Required failed: 0
- Required skipped: 1
- Score: 1.000

## Per provider

| provider | pass | fail | skip |
|---|---:|---:|---:|
| anthropic | 4 | 0 | 0 |
| gemini | 4 | 0 | 0 |
| openai | 4 | 0 | 0 |
| shared | 2 | 0 | 1 |

## Tests

| id | provider | probe | required | status | details |
|---|---|---|---:|---|---|
| openai.complete.fixture | openai | fixture_complete | true | pass | normalized response parsed |
| anthropic.complete.fixture | anthropic | fixture_complete | true | pass | normalized response parsed |
| gemini.complete.fixture | gemini | fixture_complete | true | pass | normalized response parsed |
| openai.stream.fixture | openai | fixture_stream | true | pass | events=start,delta,end |
| anthropic.stream.fixture | anthropic | fixture_stream | true | pass | events=start,delta,end |
| gemini.stream.fixture | gemini | fixture_stream | true | pass | events=delta |
| openai.tool.fixture | openai | fixture_tool_call | true | pass | tool call normalized |
| openai.extended.fixture | openai | openai_extended | true | pass | openai extended endpoints normalized |
| anthropic.tool.fixture | anthropic | fixture_tool_call | true | pass | tool call normalized |
| anthropic.extended.fixture | anthropic | anthropic_extended | true | pass | anthropic files+batches normalized |
| gemini.tool.fixture | gemini | fixture_tool_call | true | pass | tool call normalized |
| gemini.extended.fixture | gemini | gemini_extended | true | pass | gemini extended endpoints normalized |
| errors.mapping.fixture | shared | error_mapping | true | pass | http error mapping matches taxonomy |
| adapter.contract.fixture | shared | adapter_contract | true | pass | all adapters expose full plugin contract |
| transport.pycurl_stream.fixture | shared | transport_streaming | true | skip | pycurl not installed |
