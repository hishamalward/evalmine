# The demo GIF, recorded with charmbracelet/vhs (https://github.com/charmbracelet/vhs).
#
#   brew install vhs
#   cd <repo root> && vhs docs/demo.tape
#
# Everything it shows is a real run: `evalmine validate`, then the example
# suite against the deterministic fake adapter (no key, no network, no spend),
# then the calibration and win-rate sections of the report it just wrote. The
# hidden setup below only activates the venv, clears the cache so the run is
# live, and shortens the prompt. Nothing is typed for effect.

Output docs/demo.gif

Set Shell bash
Set FontSize 15
Set Width 1280
Set Height 780
Set Padding 24
Set TypingSpeed 25ms

Hide
Type "export PS1='$ '" Enter
Type "source .venv/bin/activate" Enter
Type "rm -rf .evalmine-cache reports" Enter
Type "clear" Enter
Sleep 1s
Show

Sleep 800ms
Type "evalmine validate examples/everyday-eight.yaml" Sleep 400ms Enter
Sleep 3s

Type "evalmine run examples/everyday-eight.yaml --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash --fake" Sleep 400ms Enter
Sleep 6s

Type "clear" Enter
Type "sed -n '13,40p' reports/everyday-eight/*/report.md" Sleep 400ms Enter
Sleep 9s
