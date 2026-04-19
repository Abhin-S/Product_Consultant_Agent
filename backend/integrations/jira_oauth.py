from fastapi import HTTPException, status


def jira_oauth_stub() -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={"message": "OAuth flow not yet available. Use manual token entry."},
    )