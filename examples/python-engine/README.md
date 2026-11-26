# A Python engine speaking mwip

[engine.py](engine.py) is a minimal inference engine that plugs into
ModelWeave through mwip, the engine protocol
([docs/mwip.md](../../docs/mwip.md)). It is written against stubs generated
from the proto files:

- one process is one instance: one model in one role (unified);
- it serves the mwip Engine service (`Generate`, `Abort`, `GetState`);
- it registers with the gateway's Registry by heartbeat, every `ttl_ms / 3`,
  and deregisters on exit; a rejected heartbeat prints the gateway's reason;
- instead of a model, `generate` yields a canned answer, split the way an
  engine's output parsers split real output (reasoning first when the prompt
  opened it, then content), and the stream ends with the usage.

Replace `generate` with your engine: run `request.token_ids` with
`request.inference_options` and parse the output with
`request.parsing_options`.

## Run it against the local stack

```sh
pip install -r requirements.txt
python -m grpc_tools.protoc -I ../../proto --python_out=gen --grpc_python_out=gen \
    ../../proto/modelweave/mwip/v1/*.proto

# The gateway (in Docker) calls the engine back on the compose network's
# host address; the stack publishes the gateway's Registry on :38082.
HOST_IP=$(docker network inspect modelweave_default -f '{{(index .IPAM.Config 0).Gateway}}')
export MW_CONTROL_TOKEN=local-control-token
python engine.py --registry 127.0.0.1:38082 --target $HOST_IP:50051 --model external-engine --id py-0 &
python engine.py --registry 127.0.0.1:38082 --target $HOST_IP:50052 --listen 0.0.0.0:50052 \
    --model external-engine --id py-1 &
```

`external-engine` is a model of the local stack without instances of its
own ([deploy/compose/gateway.yaml](../../deploy/compose/gateway.yaml)). Ask
the gateway for it; round robin alternates between the two instances:

```sh
curl -s localhost:38080/v1/chat/completions -H 'Authorization: Bearer sk-team-a' \
  -H 'Content-Type: application/json' \
  -d '{"model":"external-engine","messages":[{"role":"user","content":"Hello"}]}'
curl -s -H 'X-ModelWeave-Token: local-control-token' localhost:38080/admin/workers
```

Stop one with `kill %1`: it deregisters, and the other serves alone. Stop
one with `kill -9 %2` instead: its heartbeats stop, and the gateway removes
it once `discovery.heartbeat_ttl` (6s in the stack) has passed.
