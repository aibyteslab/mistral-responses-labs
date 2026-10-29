# Support agent on Mistral's Responses API

Companion code for the video *Switch From OpenAI to Mistral: Responses API Agent in 20 Lines*.
A tool-calling agent that looks up fake bookshop orders, checks the refund policy, and files a
JSON ticket, built on Mistral's Responses API (public preview, announced 28 October 2026).

All data is invented. No real customers, orders or keys.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # mistralai==3.6.0, the version this was tested with
export MISTRAL_API_KEY="..."        # never put the key in a file

python agent.py --demo agent        # the loop until the model answers
python agent.py --demo ticket       # adds the create_ticket tool, prints the ticket JSON
python agent.py --demo reasoning    # reasoning effort high, reasoning items shown
python agent.py --demo compare --models mistral-large-4,mistral-small-latest \
    --price mistral-large-4=IN,OUT --price mistral-small-latest=IN,OUT --runs 3
```

Prices are per million tokens; take them from each model's page on docs.mistral.ai on the day. Nothing is hard-coded.
With `--runs N` each model runs N times and the table shows the median. An API error (for example a 429 on a
capacity-limited preview model) is printed in the row instead of stopping the run.

## Third-party models and long context

```bash
# which endpoints list a model: global (no location commitment), EU, US
python agent.py --demo regions --models zai-glm-5-3-flash,deepseek-v4-1-flash-1017

# same agent on any endpoint
python agent.py --demo agent --model zai-glm-5-3-flash --server eu

# long context: a fake one-night log with three planted root causes
python make_log.py                                   # writes logs/incident.log (fixed seed)
python agent.py --demo longctx --model deepseek-v4-1-flash-1017
```

`longctx` counts the input tokens, asks for the three independent root causes, and prints which of the
planted causes (DB pool, full disk, expired TLS certificate) the answer names. The log is invented:
example.com hosts, made-up services, one harmless red-herring warning.

## What the loop does

1. Send the history in `input`, with the tool definitions.
2. Append the model's output items to the history.
3. For each `function_call` item: `json.loads(arguments)`, run the Python function, append
   `{"type": "function_call_output", "call_id": ..., "output": json.dumps(result)}`.
4. Repeat until a response has no function calls; the `message` item holds the answer.

The API is stateless: the app keeps the history. `store` must be omitted, false or null, and `previous_response_id` omitted or null.

## Docs

- https://docs.mistral.ai/studio/conversations/responses
- https://docs.mistral.ai/api/endpoint/responses

Public preview: parameters and behaviour can change. Tested with `mistralai` 3.6.0.

Powered by AI Bytes Lab™ · aibyteslab.com
