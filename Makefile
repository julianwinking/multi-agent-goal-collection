### Multi-Agent Goal Collection Planner ###

.PHONY: install run clean test help

# Install dependencies using poetry
install:
	poetry lock --no-update
	poetry install

# Run the multi-agent planner
run:
	poetry run python run.py

# Run with specific scenario
run-scenario:
	poetry run python src/main.py --scenario $(SCENARIO)

# Run all scenarios
run-all:
	poetry run python src/main.py --all-scenarios

# Clean up generated files and cache
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf out/

# Run tests (if you add tests later)
test:
	poetry run pytest tests/

# Update base Docker image (run from outside devcontainer)
CURRENT_BASE = pdm4ar2025:3.11-bullseye
update-base-image:
	docker pull idscfrazz/$(CURRENT_BASE)

# Help command
help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make run            - Run default scenarios"
	@echo "  make run-scenario   - Run specific scenario: make run-scenario SCENARIO=config_1.yaml"
	@echo "  make run-all        - Run all scenarios"
	@echo "  make clean          - Clean up cache and generated files"
	@echo "  make test           - Run tests"

