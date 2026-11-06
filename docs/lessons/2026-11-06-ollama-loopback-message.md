# "Start Ollama" is the wrong fix when localhost means the container

A connection refusal against a loopback Ollama URL used to end with
"Start Ollama and retry." Inside the preview container, localhost is the
container itself, so that advice sent the owner to restart a healthy host
service. Both loopback error paths (startup model-list preflight and the
stage connection-error translation) now keep the original advice and add the
container case: set OLLAMA_BASE_URL=http://host.docker.internal:11434/v1 for
native Mac Ollama. Non-loopback URLs are unchanged.
