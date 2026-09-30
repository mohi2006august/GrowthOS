.PHONY: install sim dev api worker web test lint seed

install:
	uv sync && pnpm i

# Run the agent end to end against the simulated ad environment.
sim:
	uv run growthos sim

# Web + API + workers.
dev:
	uv run growthos dev

api:
	uv run growthos api

worker:
	uv run growthos worker

web:
	pnpm --dir web dev

seed:
	uv run growthos seed

test:
	uv run pytest

lint:
	uv run ruff check growthos tests
