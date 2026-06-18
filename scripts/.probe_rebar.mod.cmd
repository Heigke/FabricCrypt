savedcmd_probe_rebar.mod := printf '%s\n'   probe_rebar.o | awk '!x[$$0]++ { print("./"$$0) }' > probe_rebar.mod
