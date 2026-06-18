savedcmd_probe_ic_write.mod := printf '%s\n'   probe_ic_write.o | awk '!x[$$0]++ { print("./"$$0) }' > probe_ic_write.mod
