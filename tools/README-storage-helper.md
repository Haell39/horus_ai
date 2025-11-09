# Storage helper (PowerShell)

This small helper lets you persist a storage configuration on a locally-running backend during development.

Files

- `set-storage-config.ps1`: PowerShell script that posts to `/api/v1/admin/storage-config`.

Usage examples (PowerShell):

- Set local mode and path:
  .\set-storage-config.ps1 -Mode local -LocalPath 'D:\\Documents' -ServerUrl 'http://localhost:8000'

- Set OneDrive mode with link:
  .\set-storage-config.ps1 -Mode onedrive -OneDriveLink 'https://onedrive.live.com/...' -ServerUrl 'http://localhost:8000'

Notes

- The backend must be reachable from the machine running this script. Default server URL is `http://localhost:8000`.
- This is intended for local development only. In production you should protect the admin endpoints.
