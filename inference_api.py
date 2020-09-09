import base64
import json
import numpy as np
from werkzeug.wrappers import Request, Response
import forward


def decode_audio(audio_bytes):
    return np.frombuffer(base64.b64decode(audio_bytes), dtype="float32")


def make_app(predict_func):
    def app(environ, start_response):
        inputs = json.loads(Request(environ).get_data())

        outputs = []
        for inp in inputs:
            try:
                pred = predict_func(decode_audio(inp))
            except Exception as e:
                print(f"Error during VAD for input {len(outputs)}: {e}")
                pred = None
            outputs.append(pred)

        return Response(json.dumps(outputs))(environ, start_response)

    return app


if __name__ == "__main__":
    import argparse
    import functools
    from werkzeug.serving import run_simple

    parser = argparse.ArgumentParser(
        description="Run simple JSON api server to predict speaker count"
    )
    parser.add_argument("--model", default="gpvf", help="model name")
    args = parser.parse_args()

    model, model_resolution, encoder = forward.load_model(args.model)
    threshold = (0.5, 0.1)

    app = make_app(
        lambda data: (
            model_resolution,
            [
                (label, int(onset), int(offset))
                for label, onset, offset in forward.predict(
                    model,
                    encoder,
                    threshold,
                    forward.extract_feature(data, **forward.LMS_ARGS)[None],
                )[0]
            ],
        )
    )
    run_simple("0.0.0.0", 5000, app, use_debugger=True)
