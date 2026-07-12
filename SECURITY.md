# Security Policy

## Current dependency advisories

**Status:** Dependency upgrades were applied on 2026-06-20 and the resulting
direct versions are pinned in `requirements.txt`. The Dependabot alert state
must still be re-scanned and verified before any alert is considered closed.
No framework components were changed.

The alert titles and numbers below are the source of record for the affected
dependency state. Advisory applicability and fixed versions must be confirmed
against the upstream advisory before an upgrade is merged.

### High severity

| Package | Alert | Summary | Interim control |
| --- | --- | --- | --- |
| `urllib3` | #45 | Decompression-bomb safeguards bypassed in parts of the streaming API | Do not stream or decompress untrusted HTTP responses without enforcing an application-level byte limit. |
| `urllib3` | #18 | Decompression-bomb safeguards bypassed when following HTTP redirects (streaming API) | Disable or tightly limit redirects for untrusted URLs; apply response-size limits across every redirect hop. |
| `urllib3` | #46 | Sensitive headers forwarded across origins in proxied low-level redirects | Do not attach credentials or other sensitive headers to requests that can redirect across origins. |
| `Pillow` | #38 | FITS GZIP decompression bomb | Treat untrusted FITS/GZIP images as unsupported until the dependency is updated; cap upload and processing resources. |
| `Pillow` | #42 | OOB write with invalid PSD tile extents (integer overflow) | Do not process untrusted PSD files. |
| `yt-dlp` | #59 | Dangerous file type creation through insufficient filename sanitization | Download only from trusted sources and save into a dedicated, non-executable directory. |
| `yt-dlp` | #60 | Arbitrary code execution via manifest downloads with `aria2c` | Do not use `aria2c` as the external downloader for untrusted manifests. |
| `yt-dlp` | #61 | Command injection when `--exec` is used | Do not use `--exec`; never form a shell command from media metadata or user-provided values. |

### Moderate severity

| Package | Alert(s) | Summary | Interim control |
| --- | --- | --- | --- |
| `aiohttp` | #27 | Unlimited trailer headers may cause uncapped memory use | Enforce request-size and concurrency limits at the reverse proxy/application boundary. |
| `aiohttp` | #57 | Incomplete WebSocket frame payloads can bypass memory limits | Do not expose WebSocket endpoints to untrusted traffic without connection, message-size, and rate limits. |
| `aiohttp` | #54 | HTTP/1 pipelined request queue is unbounded | Limit concurrent connections and request rates at the edge. |
| `aiohttp` | #53 | Unread compressed request bodies can bypass `client_max_size` during cleanup | Reject or restrict compressed request bodies from untrusted clients and enforce proxy body-size limits. |
| `aiohttp` | #52 | C HTTP parser can bypass `max_line_size` for fragmented lines | Enforce header and request-line limits at the reverse proxy. |
| `aiohttp` | #48 | Per-request cookies can cross origins on redirects | Do not use per-request authentication cookies with untrusted redirect targets. |
| `aiohttp` | #30 | Windows static-resource handler can allow UNC SSRF, NTLMv2 credential theft, or local-file reads | On Windows, do not map untrusted paths through `add_static`; reject UNC and traversal paths. |
| `aiohttp` | #31 | Multipart header-size bypass | Restrict multipart uploads from untrusted clients; apply proxy request-size limits. |
| `aiohttp` | #47 | Deserialization of untrusted data | Do not deserialize untrusted Python objects; use validated JSON or other safe data formats. |
| `aiohttp` | #36 | Duplicate `Host` headers accepted | Validate the expected host at the reverse proxy/application boundary. |
| `aiohttp` | #51 | Digest authentication credentials may be applied to cross-origin redirect challenges | Do not use digest authentication with untrusted redirect targets. |
| `pytest` | #39 | Vulnerable temporary-directory handling | Run tests only in trusted workspaces; do not run tests with elevated privileges. |
| `python-dotenv` | #40 | `set_key` can follow symlinks during cross-device rename fallback | Do not call `set_key` on paths writable by untrusted users; protect `.env` paths from symlink manipulation. |
| `yt-dlp` | #58 | Downloader cookie leak with `curl` | Do not pass sensitive cookies to `curl` for downloads that can redirect or use untrusted hosts. |
| `Pillow` | #43 | Integer overflow while processing fonts | Do not process untrusted font files. |
| `Pillow` | #41 | Heap buffer overflow with nested list coordinates | Treat untrusted image content as untrusted input and process it in a resource-constrained worker where practical. |
| `Pillow` | #44 | PDF parsing trailer infinite loop (denial of service) | Do not process untrusted PDF files with Pillow; apply processing timeouts. |

## Required remediation process

1. Review the exact advisory, affected version range, fixed version, and
   exploit preconditions for each open alert.
2. Update dependencies in a dedicated change; do not combine this work with
   framework migrations or unrelated feature changes. This upgrade was
   completed on 2026-06-20.
3. Run the project's test suite and targeted smoke tests for HTTP clients,
   media processing, and download workflows.
4. Re-scan the dependency manifest and close only alerts whose fixed version is
   installed and whose advisory has been verified as resolved.
5. Remove or revise the relevant interim control in this file when remediation
   is complete.

## Reporting a vulnerability

Do not open a public issue for a suspected undisclosed vulnerability. Report it
privately to the project maintainers with affected version(s), reproduction
steps, impact, and any suggested remediation. Do not include credentials,
tokens, or other secrets in the report.
