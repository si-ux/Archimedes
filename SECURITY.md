# Security policy

## Supported versions

Archimedes is developed on `main`. Security fixes are made there; there are no maintained release branches yet.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Report it privately through GitHub: go to the repository's **Security** tab and choose **Report a vulnerability**.
Include:

- what the problem is and which component it affects (web server, console, Workbench, DOLFINx bridge, …);
- the steps or a proof of concept to reproduce it;
- the impact you expect (for example remote code execution or data exposure).

You should get an acknowledgement within a week. We will keep you informed while we work on a fix and credit you in
the release notes unless you prefer to stay anonymous.

## Deployment model: read this before exposing the server

Archimedes is designed to run **on your own machine or in your own Codespace**. Several features are powerful by
design and are not safe on a public network:

| Component | Why it must stay private |
|---|---|
| Python console (`/ws`, `console` messages) | Runs Python commands and imported scripts on the server, with the server's permissions. |
| FE solve endpoint (`POST /api/fe/solve`) | Accepts arbitrarily large meshes; a big request can use a lot of CPU time and memory. |
| DOLFINx bridge (`ws://127.0.0.1:8791`) | Runs whatever study it receives. `docker-compose.yml` binds it to `127.0.0.1` on purpose. |

So:

- Bind the server to `localhost` when you run it on a shared machine:
  `python -m uvicorn webdemo.server:app --host 127.0.0.1 --port 8000`.
- In Codespaces, keep the forwarded port **private** (the default). Do not make it public.
- Do not put the server on the internet without adding authentication and removing or sandboxing the console.

## Privacy

Camera frames are processed in your browser by MediaPipe and are never sent to the server. Only hand landmark
coordinates (about 1 KB per frame) are sent over the WebSocket. Calibration samples and recorded training clips are
written to the local `data/` folder, which is git-ignored.
