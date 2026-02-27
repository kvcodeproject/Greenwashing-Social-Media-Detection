# Greenwashing Social Media Detection

## Project Overview
This project aims to develop a framework for detecting greenwashing in social media content. Greenwashing is the practice of misleading consumers about environmental efforts of a company or organization. As more consumers are concerned about climate change and sustainability, it is crucial to identify when companies misrepresent their environmental actions.

ESG Category Scores:
•	Boolean data: Yes=1 (positive), No/Null=0; reversed for negative polarity. Defaults: 0 for most, 1 for specific (e.g., no operations in critical countries).
•	Numeric data: Only if reported; polarity determines if higher is better.
•	Industry-specific irrelevance marked N/R.
•	Percentile rank: (worse companies + (same/2)) / total with value. Benchmarked by TRBC industry for E/S/controversies, country for G.
Materiality Matrix:
•	Dynamic weights (1-10) based on industry impact, using proxies (e.g., CO2 emissions for Emission category).
•	Methods: Industry median for numeric (relative contribution to ESG universe); transparency % for Boolean.
•	Community fixed at 5 (equal importance). Governance based on data point count.
•	Normalized to 0-100%. Updated annually for active years.


## Documentation
- **Introduction:**  Overview of greenwashing and its implications in social media.
- **Methods:**  Detailed description of the methods used for detection, including data collection and analysis techniques.
- **Usage:**  Instructions on how to use the detection framework.
- **Contributing:**  Guidelines for contributing to the project.
- **License:**  Information about the project's licensing.

## Getting Started
To start using this project:
1. Clone the repository.
2. Install the required dependencies.
3. Run the detection scripts as per the documentation.

## Contact
For inquiries, please contact the project maintainers through GitHub issues or emails.
