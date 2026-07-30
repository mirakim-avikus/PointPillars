# certs/

Place your network's TLS-inspecting CA certificate here as `HD_Groups_CA.crt` before running
`docker build`, if your network intercepts outbound HTTPS (e.g. Avikus's Prisma Access SASE
gateway, which presents a certificate issued by `CN=HD_Groups_CA`). That exact filename is
gitignored - the cert is never committed, since it's specific to whichever network you're
building on. (`00-placeholder.crt` in this directory is an unrelated tracked empty file that
keeps the Dockerfile's `COPY certs/*.crt` working even when no real cert is present - leave it
alone.)

## Do you need this?

Try building the image first (`docker build -t custom-open3d-python-cu111 .` from the repo
root, per the main [README](../README.md#installation)). If it fails during `apt-get update`
or `pip install` with an error like:

```
SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
failed: self signed certificate in certificate chain ...
```

your network is TLS-inspecting outbound traffic and you need to add the CA cert here.

## How to get the cert

From a machine on the affected network:

```bash
echo | openssl s_client -connect download.pytorch.org:443 -servername download.pytorch.org -showcerts 2>/dev/null \
  | awk 'BEGIN{c=0} /-----BEGIN CERTIFICATE-----/{c++} {print > ("cert_" c ".pem")}'
```

This splits the presented certificate chain into `cert_1.pem` (the leaf, e.g. `pytorch.org`)
and `cert_2.pem` (the CA that signed it). Confirm `cert_2.pem` is self-signed (its `issuer`
and `subject` match):

```bash
openssl x509 -in cert_2.pem -noout -subject -issuer
```

Save it here as `certs/HD_Groups_CA.crt`:

```bash
cp cert_2.pem certs/HD_Groups_CA.crt
```

Then `docker build` again from the repo root.

## Why the Dockerfile needs this

The Dockerfile copies this cert into the image's trust store (`update-ca-certificates`) and
also points `pip` at it explicitly via `PIP_CERT`/`REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE` env
vars, since `pip` bundles its own CA store (`certifi`) and doesn't use the OS trust store by
default. On a network without TLS inspection, none of this is needed - `git clone` and
`docker build` will just work without ever touching this directory.
