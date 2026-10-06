// Python console panel: a REPL on the server's ModelSession, like the Abaqus command line.
// Lines go to the server one at a time ("..." while a block is open); imported files and
// multi-line pastes run as scripts. Gesture and mouse edits show up as journal lines.

const COMMANDS = ["beam(", "column(", "round_beam(", "round_column(", "material(", "demo(", "increments(",
  "fixed(", "pinned(", "roller(", "point(", "uniform(", "trapezoidal(", "clear(", "undo()", "solve()",
  "fields()", "field(", "peak(", "at(", "reactions()", "modes()", "summary()", "show(", "cut(", "uncut()",
  "view(", "journal()", "run(", "help(", "model", "results"];

export function createConsole(send) {
  const $ = (id) => document.getElementById(id);
  const panel = $("console"), out = $("con-out"), inp = $("con-in"), ps = $("con-ps"), vp = $("viewport");
  const history = []; let hIdx = 0, more = false, busy = 0, lastJournal = 0, journal = [];
  try { history.push(...JSON.parse(localStorage.getItem("archimedes-con-history") || "[]")); hIdx = history.length; } catch (e) { /* private mode */ }

  function print(text, cls = "") {
    if (!text) return;
    const span = document.createElement("span");
    if (cls) span.className = cls;
    span.textContent = text.endsWith("\n") ? text : text + "\n";
    out.appendChild(span);
    while (out.childNodes.length > 2000) out.removeChild(out.firstChild);
    out.scrollTop = out.scrollHeight;
  }
  function setBusy(d) { busy = Math.max(0, busy + d); panel.classList.toggle("busy", busy > 0); }
  function open(on = panel.classList.contains("hidden")) {
    panel.classList.toggle("hidden", !on); vp.classList.toggle("console-open", on);
    $("btn-console").classList.toggle("on", on);
    if (on) { inp.focus(); out.scrollTop = out.scrollHeight; }
  }
  function submit(line) {
    print((more ? "... " : ">>> ") + line, "echo");
    if (line.trim()) history.push(line);
    if (history.length > 300) history.splice(0, history.length - 300);
    hIdx = history.length;
    try { localStorage.setItem("archimedes-con-history", JSON.stringify(history.slice(-100))); } catch (e) { /* ignore */ }
    setBusy(1); send({ type: "console", line });
  }
  function runScript(name, code) { print(`# import ${name}`, "note"); setBusy(1); send({ type: "console_script", name, code }); }

  inp.addEventListener("keydown", (e) => {
    e.stopPropagation();                                   // keep viewport shortcuts (R, X, U...) out of the console
    if (e.key === "Enter") { e.preventDefault(); const l = inp.value; inp.value = ""; submit(l); }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (hIdx > 0) inp.value = history[--hIdx]; }
    else if (e.key === "ArrowDown") { e.preventDefault(); hIdx = Math.min(history.length, hIdx + 1); inp.value = history[hIdx] || ""; }
    else if (e.key === "Tab") {
      e.preventDefault();
      const pos = inp.selectionStart, before = inp.value.slice(0, pos), m = before.match(/[A-Za-z_]\w*$/);
      const cands = m ? COMMANDS.filter((c) => c.startsWith(m[0])) : [];
      let ins = "    ";
      if (cands.length === 1) ins = cands[0].slice(m[0].length);
      else if (cands.length > 1) { print(cands.join("  "), "note"); ins = commonPrefix(cands).slice(m[0].length); }
      inp.value = before + ins + inp.value.slice(pos); inp.selectionStart = inp.selectionEnd = pos + ins.length;
    } else if ((e.key === "c" && e.ctrlKey && !inp.value.length) || (e.key === "Escape" && (more || inp.value))) {
      e.preventDefault(); inp.value = ""; if (more) { print("KeyboardInterrupt", "err"); more = false; ps.textContent = ">>>"; send({ type: "console_reset" }); }
    } else if (e.key === "Escape") open(false);
    else if (e.key === "l" && e.ctrlKey) { e.preventDefault(); out.textContent = ""; }
  });
  inp.addEventListener("paste", (e) => {
    const text = e.clipboardData.getData("text");
    if (!text.includes("\n")) return;
    e.preventDefault();
    for (const l of text.replace(/\r/g, "").split("\n")) print("... " + l, "echo");
    runScript("<paste>", text);
  });
  $("con-import").onclick = () => $("con-file").click();
  $("con-file").onchange = async (e) => {
    for (const f of e.target.files) runScript(f.name, await f.text());
    e.target.value = "";
  };
  $("con-journal").onclick = () => {
    const body = ["# Archimedes journal: every model edit this session, from gestures, the mouse and the console.",
      "# Run it with Import script... or run('this_file.py') from fe_scripts/.", "",
      ...journal.map((j) => j.cmd + (j.src === "gui" ? "  # gesture / mouse" : "")), ""].join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([body], { type: "text/x-python" }));
    a.download = "archimedes_journal.py"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $("con-clear").onclick = () => { out.textContent = ""; inp.focus(); };
  $("con-close").onclick = () => open(false);
  $("btn-console").onclick = () => open();
  panel.addEventListener("click", (e) => { if (e.target === out && !getSelection().toString()) inp.focus(); });

  print("Archimedes Python console · help() lists the commands · Tab completes · ↑↓ history", "note");
  print("e.g.  beam(b=300, h=500, L=6000); pinned(at=0); roller(at=1); uniform(10); solve(); peak('U3')", "note");

  return {
    open,
    onOutput(m) {
      setBusy(-1);
      more = !!m.more; ps.textContent = more ? "..." : ">>>";
      print(m.output, m.error ? "err" : "");
    },
    // journal entries from the server; gesture / mouse ones are echoed so you can learn the commands
    onJournal(entries) {
      if (!entries) return;
      journal = journal.filter((j) => j.n < (entries[0] ? entries[0].n : Infinity)).concat(entries);
      for (const j of entries) {
        if (j.n <= lastJournal) continue;
        if (lastJournal && j.src === "gui") print(j.cmd + "   # from gesture / mouse", "note");
      }
      if (entries.length) lastJournal = entries[entries.length - 1].n;
    },
  };
}

function commonPrefix(a) {
  let p = a[0];
  for (const s of a) while (!s.startsWith(p)) p = p.slice(0, -1);
  return p;
}
