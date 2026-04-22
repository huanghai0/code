# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a personal learning repository focused on dynamic programming algorithms and data structures. The repository contains Java implementations with corresponding documentation in Chinese.

## Project Structure

```
code/
├── src/
│   ├── dp/           # Dynamic programming implementations (Java)
│   │   ├── ZeroOneKnapsack.java
│   │   ├── Fibonacci.java
│   │   ├── HouseRobber.java
│   │   ├── MaximumSubarray.java
│   │   └── BestTimeToBuyAndSellStockII.java
│   └── *.java        # Other algorithm implementations
└── doc/              # Algorithm documentation (Markdown)
    ├── 01-knapsack.md
    └── 02-dynamic-programming-principle.md
```

## Running Code

### Java Compilation and Execution

Each Java file can be compiled and run individually:

```bash
# Compile
javac src/dp/ZeroOneKnapsack.java

# Run
java src.dp.ZeroOneKnapsack
```

Or compile all files in a directory:

```bash
# Compile all Java files
javac src/dp/*.java src/*.java
```

Note: Some files have `main()` but not `public static void main(String[] args)`. Check the method signature before running.

## Code Organization

### Package Structure
- All DP implementations use `package src.dp;`
- Root-level Java files (e.g., `JumpGameII.java`, `MaxSlidingWindow.java`) are not in a package

### Documentation Conventions
- Each algorithm has comprehensive documentation in `doc/` folder
- Documentation includes problem definition, state transition equations, complexity analysis, and code examples in multiple languages (Python, JavaScript, Java, C++)
- Code comments are in Chinese

### Implementation Patterns
- DP solutions typically include both 2D and 1D optimized versions
- 1D DP for 01背包 must iterate capacity in **reverse order** (`for j = capacity; j >= weights[i]; j--`) to avoid using the same item multiple times
- Space optimization is prioritized where applicable
- Test cases are included in `main()` methods or as separate test blocks

## Common Issues

1. **Package declarations**: Files in `src/dp/` have `package src.dp;` which affects compilation paths
2. **Main method signatures**: Some files use `static void main()` instead of `public static void main(String[] args)` - these won't run as standalone programs
3. **IDE files**: `.idea`, `.iml`, `.iws` files are ignored (IntelliJ IDEA specific)
