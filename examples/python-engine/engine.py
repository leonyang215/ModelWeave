"""A minimal inference engine speaking mwip, ModelWeave's engine protocol.

One process is one instance: one model in one role (unified here). It
serves the mwip Engine service and registers with the gateway's Registry by
heartbeat until it stops, then deregisters. Instead of running a model it
streams a canned answer, split the way an engine's output parsers split
real output: reasoning first when the prompt opened it, then content, then
the usage. Replace `generate` with your engine.

Generate the stubs first (see README.md), then:

    python engine.py --registry 127.0.0.1:38082 --target 172.18.0.1:50051 \\
        --model external-engine --id py-0
"""

import argparse
import asyncio
import os
import signal
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen"))

import grpc  # noqa: E402

from modelweave.mwip.v1 import (  # noqa: E402
    engine_pb2,
    engine_pb2_grpc,
    registry_pb2,
    registry_pb2_grpc,
    types_pb2,
)

MAX_MESSAGE = 64 << 20  # mwip messages can be large: accept at least 64 MiB


def generate(instance_id: str, request: engine_pb2.GenerateRequest):
    """The engine: yields (reasoning, content) pieces for the request.

    A real engine runs request.token_ids with request.inference_options and
    parses its output with request.parsing_options (reasoning_parser,
    tool_call_parser, stop_sequences).
    """
    stage = request.parsing_options.thinking_initial_stage
    if stage == types_pb2.THINKING_STAGE_REASONING_START:
        yield "The prompt has %d tokens; " % len(request.token_ids), ""
        yield "I will answer briefly.", ""
    answer = "Hello from the Python engine %s." % instance_id
    for word in answer.split(" "):
        yield "", word + " "


class Engine(engine_pb2_grpc.EngineServicer):
    """The Engine service of this instance."""

    def __init__(self, instance_id):
        self.id = instance_id
        self.running = {}  # request_id -> the task streaming it
        self.state = types_pb2.EngineState(max_running_requests=8)

    async def Generate(self, request, context):
        if request.disagg.mode == engine_pb2.DISAGG_MODE_SPLIT:
            # This engine runs whole requests only (a unified instance).
            yield engine_pb2.GenerateResponse(error=engine_pb2.Error(
                code=engine_pb2.ERROR_CODE_BAD_REQUEST, message="split legs are not supported", retriable=False))
            return
        self.running[request.request_id] = asyncio.current_task()
        self.state.running_requests += 1
        try:
            reasoning_tokens = completion_tokens = 0
            for reasoning, content in generate(self.id, request):
                completion_tokens += 1
                reasoning_tokens += 1 if reasoning else 0
                yield engine_pb2.GenerateResponse(delta=engine_pb2.Delta(
                    reasoning=reasoning, content=content, token_ids=[completion_tokens]))
                await asyncio.sleep(0.005)
            yield engine_pb2.GenerateResponse(finish=engine_pb2.Finish(
                reason=engine_pb2.FINISH_REASON_STOP,
                usage=engine_pb2.Usage(prompt_tokens=len(request.token_ids), completion_tokens=completion_tokens,
                                       reasoning_tokens=reasoning_tokens)))
        finally:
            # A canceled call ends here too: free the request.
            self.state.running_requests -= 1
            self.running.pop(request.request_id, None)

    async def Abort(self, request, context):
        aborted = 0
        for rid in request.request_ids:
            task = self.running.pop(rid, None)
            if task is not None:
                task.cancel()
                aborted += 1
        return engine_pb2.AbortResponse(aborted=aborted)

    async def GetState(self, request, context):
        return engine_pb2.GetStateResponse(state=self.state)


async def register(args, engine, stop):
    """Heartbeats until stop (well within the gateway's TTL), then
    deregisters. A rejected heartbeat says why; the instance retries."""
    metadata = (("x-modelweave-token", args.token),) if args.token else ()
    instance = registry_pb2.Instance(id=args.id, model=args.model, role=types_pb2.ROLE_UNIFIED,
                                     target=args.target, max_running_requests=8)
    async with grpc.aio.insecure_channel(args.registry) as channel:
        registry = registry_pb2_grpc.RegistryStub(channel)
        while not stop.is_set():
            try:
                resp = await registry.Heartbeat(registry_pb2.HeartbeatRequest(instance=instance, state=engine.state),
                                                metadata=metadata, timeout=2)
                if resp.registered:
                    print("registered %s (heartbeat ttl %d ms)" % (args.id, resp.ttl_ms), flush=True)
                interval = max(resp.ttl_ms / 3000, 0.5)
            except grpc.aio.AioRpcError as err:
                print("heartbeat rejected: %s: %s" % (err.code().name, err.details()), flush=True)
                interval = 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
        resp = await registry.Deregister(registry_pb2.DeregisterRequest(instance_id=args.id),
                                         metadata=metadata, timeout=2)
        print("deregistered %s (removed: %s)" % (args.id, resp.removed), flush=True)


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--listen", default="0.0.0.0:50051", help="address of the Engine service")
    parser.add_argument("--target", required=True, help="mwip target the gateway calls (host:port)")
    parser.add_argument("--registry", required=True, help="the gateway's mwip Registry (host:port)")
    parser.add_argument("--token", default=os.environ.get("MW_CONTROL_TOKEN", ""), help="control token")
    parser.add_argument("--model", required=True, help="the gateway's model name")
    parser.add_argument("--id", default="py-0", help="instance id, unique among the gateway's instances")
    args = parser.parse_args()

    engine = Engine(args.id)
    server = grpc.aio.server(options=[("grpc.max_receive_message_length", MAX_MESSAGE),
                                      ("grpc.max_send_message_length", MAX_MESSAGE)])
    engine_pb2_grpc.add_EngineServicer_to_server(engine, server)
    server.add_insecure_port(args.listen)
    await server.start()
    print("serving %s on %s" % (args.id, args.listen), flush=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await register(args, engine, stop)
    await server.stop(grace=2)


if __name__ == "__main__":
    asyncio.run(main())
