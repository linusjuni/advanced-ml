# Advanced Machine Learning — DTU

Monorepo for miniprojects and weekly exercises.

## Structure

```plaintext
miniproject1/   # Miniproject 1
miniproject2/   # Miniproject 2
miniproject3/   # Miniproject 3
weekly/         # Weekly exercises
```

## Setup

```bash
uv sync
```

## Adding dependencies

```bash
# To a miniproject
uv add <package> --package miniproject1

# To root (for weekly exercises)
uv add <package>
```
