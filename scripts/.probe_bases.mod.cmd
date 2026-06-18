savedcmd_probe_bases.mod := printf '%s\n'   probe_bases.o | awk '!x[$$0]++ { print("./"$$0) }' > probe_bases.mod
