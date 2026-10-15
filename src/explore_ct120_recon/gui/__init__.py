"""Browser front end for the reconstruction drivers.

``ct120-gui`` serves a small web application from the machine that owns the
GPUs. Every job it starts is one of the package's own console scripts run as a
subprocess, so a job started from the page is reproducible from the terminal:
the job record keeps the exact command line, the machine it ran on, the log
and the result path.

Modules:
  schema     the drivers the GUI knows, their argparse parsers turned into a
             form description, and the reverse (form values -> command line)
  jobs       the job record and its SQLite store
  executors  where a job runs: this machine now, a cluster later
  app        the FastAPI application (JSON API + the built React page)
  cli        ``ct120-gui``

Install the extra: ``pip install "eXplore_CT_120_reconstruction_algorithms[gui]"``.
"""
