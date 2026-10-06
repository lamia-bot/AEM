from mea_analyzer.runtime import configure_logging

# Configure diagnostics before importing the heavier GUI/analysis stack so even
# import-time failures in a packaged build leave a useful log for support.
configure_logging()


if __name__ == "__main__":
    from mea_analyzer.main_window import run_app

    run_app()
