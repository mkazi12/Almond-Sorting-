"""Database and media storage for DINOS (PostgreSQL + local files).

All persistence goes through this package so UI code never builds engines,
SQL, or file paths itself. That boundary is what lets a future API service
take over storage for remote clients without rewriting the UI; see
docs/database.md.
"""
