# Secure Remote Access

Remote experiment machines are accessed with SSH keys. Do not put a host
address, user name, password, private key, or `sshpass` command in this
repository.

Use portable variables for machine-local paths referenced by experiment notes:

```bash
export V2X_ROOT="<repository-root>"
export V2X_DATA_ROOT="<machine-local-data-root>"
export V2X_HOME="<machine-local-work-root>"
```

On the client machine, keep the connection details in `~/.ssh/config`:

```sshconfig
Host v2x-remote
    HostName <configured-on-your-machine>
    Port <configured-on-your-machine>
    User <configured-on-your-machine>
    IdentityFile ~/.ssh/id_ed25519_v2x_remote
    IdentitiesOnly yes
```

Verify that password authentication is not used:

```bash
ssh -o PasswordAuthentication=no \
    -o PreferredAuthentications=publickey \
    v2x-remote
```

For scripts that must not depend on a local SSH alias, configure the values in
the calling environment rather than source code or repository files:

```bash
export V2X_REMOTE_HOST="<configured-on-your-machine>"
export V2X_REMOTE_PORT="<configured-on-your-machine>"
export V2X_REMOTE_USER="<configured-on-your-machine>"
ssh -p "${V2X_REMOTE_PORT}" \
    "${V2X_REMOTE_USER}@${V2X_REMOTE_HOST}"
```

The variables identify a machine but never carry a password. Keep any
machine-specific configuration local and untracked.
