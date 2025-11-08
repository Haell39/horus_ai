from fastapi import APIRouter, HTTPException, Query
import os
import shutil

router = APIRouter()


@router.get('/admin/disk-usage')
def disk_usage(path: str = Query(..., description='Path to check disk usage')):
    """Return disk usage stats (GB) for a given path. Used by frontend to display local disk usage."""
    try:
        if not path:
            raise HTTPException(status_code=400, detail='path is required')
        # Make sure the path exists; if file given, use its mount point
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f'Path not found: {path}')
        # Resolve to the mount point / directory
        target = path
        if os.path.isfile(path):
            target = os.path.dirname(path) or path

        usage = shutil.disk_usage(target)
        total_gb = round(usage.total / (1024**3), 2)
        used_gb = round((usage.total - usage.free) / (1024**3), 2)
        free_gb = round(usage.free / (1024**3), 2)
        percent = round((used_gb / total_gb) * 100 if total_gb else 0.0, 2)
        return {
            'path': os.path.abspath(target),
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'percent': percent,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
