"""Compatibility entry point for the maintained read-only job recovery CLI."""
from runpy import run_module

if __name__ == "__main__":
    run_module("scripts.recover_ibm_job", run_name="__main__")
