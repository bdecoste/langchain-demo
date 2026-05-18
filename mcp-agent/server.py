"""Custom MCP server exposing tools for the LangChain agent."""

import math
import statistics
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("UtilityServer")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression. Supports +, -, *, /, **, sqrt, log, etc."""
    allowed = {
        "abs": abs, "round": round,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
        "pow": pow, "sum": sum, "min": min, "max": max,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def statistics_summary(numbers: list[float]) -> dict:
    """Return mean, median, std dev, min, and max for a list of numbers."""
    if not numbers:
        return {"error": "Empty list"}
    return {
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "stdev": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers),
        "max": max(numbers),
        "count": len(numbers),
    }


@mcp.tool()
def current_utc_time() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    Supported: km/miles, kg/lbs, celsius/fahrenheit, meters/feet, liters/gallons.
    """
    conversions: dict[tuple[str, str], float | None] = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = value * conversions[key]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # Temperature needs special handling
    f, t = from_unit.lower(), to_unit.lower()
    if f == "celsius" and t == "fahrenheit":
        return f"{value}°C = {value * 9/5 + 32:.2f}°F"
    if f == "fahrenheit" and t == "celsius":
        return f"{value}°F = {(value - 32) * 5/9:.2f}°C"

    return f"Unsupported conversion: {from_unit} → {to_unit}"


@mcp.tool()
def word_count(text: str) -> dict:
    """Count words, characters, sentences, and unique words in a text string."""
    words = text.split()
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    return {
        "characters": len(text),
        "words": len(words),
        "unique_words": len(set(w.lower().strip(".,!?") for w in words)),
        "sentences": len(sentences),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
