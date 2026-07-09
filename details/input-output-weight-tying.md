# Input-Output Weight Tying (Token Space Inversion)

Maps the final word token generation head directly back to the step-zero input lookup table.

## Architecture Diagram
`mermaid
flowchart TD
    A[Input] --> B[Input-Output Weight Tying (Token Space Inversion)]
    B --> C[Output]
`

[Back to README](../README.md)
