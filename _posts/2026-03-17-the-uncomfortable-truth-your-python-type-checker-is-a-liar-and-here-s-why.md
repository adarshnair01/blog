---
layout: post
title: "The Uncomfortable Truth: Your Python Type Checker Is A Liar (And Here's Why)"
date: 2026-03-17 07:47:12 +0530
excerpt: "Ever wonder if your meticulously typed Python code is truly ironclad? Prepare for a shocking revelation: the 'truth' of your type hints might depend entirely on which type checker you ask. We dive deep into the silent war for Python typing spec conformance."
author: "Adarsh Nair"
categories: development
tags: ["Python", "Typing", "Mypy", "Pyright", "Type Checkers", "PEP", "Code Quality", "Static Analysis", "Developer Tools"]
---

The Uncomfortable Truth: Your Python Type Checker Is A Liar (And Here's Why)

In the quest for robust, maintainable Python code, type hints have emerged as a beacon of clarity and a shield against common errors. We meticulously annotate our functions, classes, and variables, trusting that tools like Mypy and Pyright will guard our codebase with unwavering vigilance. We sleep soundly, confident in the static analysis that promises to catch bugs before they ever see a runtime.

But what if I told you that this confidence is, at times, misplaced? What if the "truth" of your Python type system isn't a singular, immutable reality, but rather a fluid concept, interpreted differently by the very tools sworn to uphold it?

This isn't hyperbole. This is the uncomfortable truth lurking beneath the surface of Python's thriving type-hinting ecosystem. The official "typing spec" – a collection of PEPs (Python Enhancement Proposals) – is the sacred text, but its interpretation is far from monolithic. Welcome to the silent war for Python typing spec conformance, where the champions, Mypy and Pyright, occasionally diverge, leaving developers caught in the crossfire.

By the end of this deep dive, you'll understand *why* these discrepancies exist, *where* they manifest, and *how* to navigate this nuanced landscape to truly harden your Python applications.

### The Genesis of Truth: Python's Typing Spec and Its Guardians

Before we expose the cracks, let's appreciate the foundation. Python's type hinting journey began in earnest with **PEP 484 (Type Hints)**, introducing the `typing` module and the core syntax for annotations. This was a monumental shift, bringing optional static typing to a dynamically typed language. Since then, a flurry of subsequent PEPs has refined and expanded the spec:

*   **PEP 561 (Distributing Type Information):** Defined how libraries ship type hints (the `py.typed` marker).
*   **PEP 586 (Literal Types):** Introduced `Literal` for precise value-based types (e.g., `Literal["GET", "POST"]`).
*   **PEP 612 (ParamSpec):** Revolutionized typing for higher-order functions by allowing the capture of callable parameter types.
*   **PEP 647 (TypeGuard):** Provided a way to inform type checkers about type narrowing performed by runtime checks.
*   **PEP 655 (Marking `TypedDict` items as `Required` or `NotRequired`):** Enhanced the expressiveness of `TypedDict`.

These PEPs collectively form the "typing spec." They are the blueprints, the constitution, the undeniable source of truth for how Python types *should* behave.

Enter the guardians:

1.  **Mypy:** The venerable pioneer. Developed by Jukka Lehtosalo, it's the reference implementation for PEP 484 and has been instrumental in shaping the early ecosystem. Mypy is written in Python, boasts a rich plugin system, and has a reputation for being robust and highly configurable.
2.  **Pyright:** The challenger from Microsoft. Born out of the TypeScript team's experience, Pyright is written in TypeScript and focuses on speed, correctness, and a "strict by default" philosophy. It powers Pylance, the popular Python language server in VS Code, and is increasingly integrated into other tools like Ruff.

Both tools aim to enforce the typing spec. Both are incredibly powerful. Yet, they sometimes disagree. Why? Because even a "spec" requires interpretation, especially when dealing with the inherent flexibility of Python and the evolving nature of the type system.

### The Battleground: Key Areas of Conformance Divergence

The discrepancies between Mypy and Pyright aren't about fundamental disagreements on basic types like `str` or `int`. They emerge in the nuanced corners of the type system, the edge cases, and the areas where the PEPs leave room for interpretation or where one checker has implemented a newer PEP more fully than the other.

Let's dissect some critical areas where their interpretations can lead to different "truths."

#### 1. The Elusive `None`: Implicit `Optional` and Strictness

One of the most common sources of confusion for Python developers is `None`. In many contexts, Python allows `None` where a type hint might imply a non-`None` value. The PEPs specify that `T | None` (or `Optional[T]`) should be used explicitly. However, type checkers vary in how strictly they enforce this.

Consider this example:

```python
# test_optional.py
from typing import Optional

def process_data(data: str) -> str:
    """Processes a string."""
    return data.upper()

def get_nullable_string() -> Optional[str]:
    """Might return a string or None."""
    return None

# Scenario 1: Implicit None assignment
value: str = get_nullable_string() # type: ignore [assignment] # Explicit ignore for demonstration
print(process_data(value))

# Scenario 2: Function argument with implicit None
def print_length(text: str):
    print(len(text))

maybe_text: Optional[str] = None
print_length(maybe_text) # type: ignore [arg-type] # Explicit ignore for demonstration
```

**Mypy's Behavior (default strictness):**
With default Mypy settings, `mypy test_optional.py` might report errors for both scenarios, as it expects explicit `Optional[str]` or `Union[str, None]`. However, Mypy has a `no-implicit-optional` flag. If this is *not* enabled (or if `strict_optional = False` in `mypy.ini`), Mypy can sometimes be more lenient, especially in older versions or specific configurations, allowing `None` where it might logically flow into a `str`.

**Pyright's Behavior (default strictness):**
Pyright, by default, is significantly stricter regarding `None`. It almost universally requires explicit handling of `None` through `Optional[T]` or `Union[T, None]` and will flag scenarios like the above as errors:

```
# Pyright output for test_optional.py
test_optional.py:12:13 - error: Expression of type "str | None" cannot be assigned to declared type "str"
  Type "None" cannot be assigned to type "str" (reportAssignmentType)
test_optional.py:18:14 - error: Argument of type "str | None" cannot be assigned to parameter "text" of type "str" in function "print_length"
  Type "None" cannot be assigned to type "str" (reportArgumentType)
```

**Takeaway:** Pyright's stricter adherence to `None` safety often leads to more robust code, forcing developers to explicitly handle potential `None` values, which is generally a good practice. Mypy can achieve similar strictness with the right configuration (`--no-implicit-optional` or `strict_optional = True`).

#### 2. `TypedDict` and the Dance of Keys: Strictness vs. Flexibility

`TypedDict` (introduced in PEP 589 and further refined by PEP 655) is a powerful tool for defining dictionary schemas with static type checking. It's meant to enforce specific keys and their types. But what happens when extra, undeclared keys are present, or when keys are missing?

```python
# test_typeddict.py
from typing import TypedDict, NotRequired

class UserProfile(TypedDict):
    name: str
    age: int
    email: NotRequired[str]

# Scenario 1: Missing Required Key
incomplete_profile: UserProfile = {"name": "Alice"}

# Scenario 2: Extra Key
extra_key_profile: UserProfile = {"name": "Bob", "age": 25, "city": "New York"}

# Scenario 3: Correct Profile
correct_profile: UserProfile = {"name": "Charlie", "age": 30, "email": "charlie@example.com"}
```

**Mypy's Behavior:**
By default, Mypy is relatively lenient with extra keys in `TypedDict` assignments, especially when the `TypedDict` isn't `total=True` (which it is by default). For missing *required* keys, Mypy will generally flag an error.

```
# Mypy output for test_typeddict.py (default settings)
test_typeddict.py:9: error: Missing key 'age' for TypedDict "UserProfile"
```
Mypy will likely *not* error on `extra_key_profile` by default. To make Mypy strict about extra keys, you need to use `TypedDict(..., total=True, extra_keys=Literal['never'])` (this is not standard PEP, rather a Mypy extension or specific config). Usually, `total=True` means all *declared* keys are required. Mypy's default behavior for *extra* keys is to ignore them unless explicitly configured otherwise or specified in a more complex `TypedDict` definition.

**Pyright's Behavior:**
Pyright, on the other hand, is much stricter by default. It assumes that if you define a `TypedDict`, you mean that *only* those keys are allowed.

```
# Pyright output for test_typeddict.py
test_typeddict.py:9:24 - error: TypedDict "UserProfile" missing key "age" (reportTypedDictNotRequiredAccess)
test_typeddict.py:12:24 - error: TypedDict "UserProfile" does not support item "city" (reportTypedDictNotRequiredAccess)
```
Pyright explicitly flags both missing required keys and the presence of undeclared keys. This aligns with a philosophy of maximum type safety, treating `TypedDict` as a strict schema.

**Takeaway:** Pyright's strictness with `TypedDict` provides stronger guarantees about data structure, preventing unexpected keys from creeping into your data. If you desire this level of strictness with Mypy, you'll need to research specific configuration options or plugins.

#### 3. `Protocol` Conformance: Structural Subtyping Nuances

`Protocol` (introduced in PEP 544) is Python's answer to structural subtyping – "if it walks like a duck and quacks like a duck, it's a duck." A class conforms to a protocol if it has the required methods and attributes with compatible types, regardless of inheritance.

```python
# test_protocol.py
from typing import Protocol

class Closable(Protocol):
    def close(self) -> None: ...

class FileManager:
    def close(self) -> None:
        print("File Manager closed.")

class DatabaseConnection:
    def disconnect(self) -> None:
        print("DB disconnected.")

def shutdown_resource(resource: Closable):
    resource.close()

shutdown_resource(FileManager())
shutdown_resource(DatabaseConnection()) # type: ignore [arg-type]
```

**Mypy's Behavior:**
Mypy generally handles `Protocol`s well. It will correctly identify `FileManager` as conforming to `Closable` and `DatabaseConnection` as not.

```
# Mypy output for test_protocol.py
test_protocol.py:20: error: Argument 1 to "shutdown_resource" has incompatible type "DatabaseConnection"; expected "Closable"
test_protocol.py:20: note: 'DatabaseConnection' is missing following members of protocol "Closable":
test_protocol.py:20: note:   close
```

**Pyright's Behavior:**
Pyright also implements `Protocol`s robustly and will produce similar errors.

```
# Pyright output for test_protocol.py
test_protocol.py:20:21 - error: Argument of type "DatabaseConnection" cannot be assigned to parameter "resource" of type "Closable" in function "shutdown_resource"
  Type "DatabaseConnection" is incompatible with protocol "Closable"
    "close" is not present in type "DatabaseConnection" (reportArgumentType)
```

**Takeaway:** While both handle basic `Protocol` conformance well, subtle differences can emerge with more complex scenarios, such as protocols with properties, `__init__` methods, or generic protocols. The key is that both adhere to the structural subtyping principle, but their internal algorithms for checking compatibility might have minor divergences in edge cases or performance. Generally, this is an area of strong conformance for both.

#### 4. `Any` and Untyped Code: The Escape Hatch Dilemma

`Any` is Python's "escape hatch" from strict type checking. It allows dynamic behavior and interoperability with untyped code, but it also bypasses all type safety. How type checkers treat `Any` and untyped function definitions can significantly impact the "truth" of your codebase's type safety.

```python
# test_any.py
from typing import Any

def process_anything(data: Any):
    # No type checking here for 'data'
    data.do_something_non_existent()
    return data

def untyped_function(arg): # No type hints
    return arg + 1

def typed_function(arg: int) -> int:
    return arg + 1

result_any = process_anything(123)
result_untyped = untyped_function("hello") # This will fail at runtime, but type checker's view?
result_typed = typed_function("world") # type: ignore [arg-type]
```

**Mypy's Behavior:**
Mypy, by default, will warn about calling `untyped_function` without annotations if `disallow_untyped_defs` is enabled. It will typically flag `result_typed` as an error. For `process_anything`, `Any` means it won't check the call to `do_something_non_existent()`.

```
# Mypy output for test_any.py (with --disallow-untyped-defs)
test_any.py:7: error: Function is missing a type annotation for one or more arguments
test_any.py:7: error: Function is missing a return type annotation
test_any.py:17: error: Argument 1 to "typed_function" has incompatible type "str"; expected "int"
```
Mypy's `disallow_untyped_defs` and `disallow_any_unimported` (among others) are crucial for tightening `Any`'s grip.

**Pyright's Behavior:**
Pyright, with its default strictness (`reportMissingTypeStubs`, `reportUntypedBaseClass`, `reportMissingTypeArgument`), tends to be very vocal about untyped code and implicit `Any`. It will also flag `result_typed` as an error and potentially warn about `untyped_function` if its reporting levels are configured appropriately.

```
# Pyright output for test_any.py (default strictness)
test_any.py:16:18 - error: Argument of type "str" cannot be assigned to parameter "arg" of type "int" in function "typed_function" (reportArgumentType)
```
Pyright's `reportMissingTypeStubs` and `reportUntypedFunctionPartial` can mimic Mypy's `disallow_untyped_defs` in many cases, pushing for stronger typing.

**Takeaway:** Both tools offer mechanisms to control the "Any" problem, but their default configurations and the granularity of their controls can differ. Pyright often pushes for more explicit type information by default, while Mypy allows for a more gradual adoption path with configurable strictness. The "truth" of your code's type safety is severely compromised if `Any` is used liberally and untyped code is ignored.

### Why Conformance Matters (Or Doesn't Always)

The existence of these divergences isn't necessarily a sign of failure, but rather a reflection of the challenges in defining a precise, unambiguous specification for a dynamic language, and the different philosophies of tool builders.

1.  **Developer Experience:** Switching between projects or teams using different type checkers can be jarring. Code that passes Mypy might fail Pyright, and vice versa. This can lead to frustration and "type checker wars."
2.  **Ecosystem Fragmentation:** If libraries are typed with one checker in mind, they might exhibit unexpected behavior or type errors when consumed by projects using another. This hinders the goal of a universally type-safe Python ecosystem.
3.  **Future-Proofing:** Relying on behavior specific to one type checker, especially if it deviates from the spirit of the PEPs, could lead to breaking changes if that checker aligns more closely with the spec in future versions.
4.  **Pragmatism vs. Purity:** Sometimes, a type checker might intentionally deviate or be more lenient for pragmatic reasons (e.g., to support common Python idioms that are hard to type strictly). Pyright, having learned from TypeScript, often leans towards stricter purity.

However, these divergences are often minor in the grand scheme. Both Mypy and Pyright provide immense value, catching countless bugs and improving code quality. The core `typing` module types are consistently understood. The differences usually lie in the interpretation of implicit behaviors, error reporting granularity, and the speed of implementing the very latest, most experimental PEPs.

### Choosing Your Champion (Or Wielding Both)

So, what's a developer to do?

1.  **Pick One and Configure It Strictly:** The most common approach is to choose either Mypy or Pyright and configure it to be as strict as your team can reasonably tolerate. For Mypy, this means enabling flags like `--strict`, `--no-implicit-optional`, `--disallow-untyped-defs`, etc., or using a `mypy.ini` with `[mypy]` section and `warn_unused_ignores = True`, `disallow_untyped_defs = True`, `no_implicit_optional = True`, `check_untyped_defs = True`, etc. For Pyright, many strict checks are enabled by default, but you can further fine-tune `reportMissingTypeStubs`, `reportUntypedBaseClass`, etc.

2.  **Standardize within Your Team/Org:** Ensure everyone on a project uses the same type checker and the same configuration. This prevents "it works on my machine" type errors related to static analysis.

3.  **Understand the "Why":** When you encounter an error, don't just blindly `type: ignore`. Take the time to understand *why* the type checker is flagging it. Is it a legitimate type safety issue? Is it a configuration difference? Is it a known divergence between checkers?

4.  **Consider Dual-Checking (for Libraries):** If you're building a widely used library, you might consider running both Mypy and Pyright in your CI/CD pipeline. This ensures maximum compatibility and catches potential issues that one checker might miss. This is especially useful for uncovering subtle spec interpretation differences.

5.  **Stay Informed:** The Python typing landscape is constantly evolving. Keep an eye on new PEPs, updates to Mypy and Pyright, and discussions within the community.

### Conclusion: Embracing the Nuance of Type Truth

The idea that your Python type checker might be "lying" to you isn't meant to breed distrust, but to foster a deeper, more nuanced understanding of type checking. There isn't a single, universally agreed-upon "truth" for every single corner of the Python typing spec. Instead, we have highly sophisticated tools, Mypy and Pyright, each striving to enforce the spec while balancing strictness with practicality.

By understanding their philosophical differences and how they manifest in concrete code, you can make informed decisions, configure your tools effectively, and ultimately write more robust, maintainable, and truly type-safe Python code. The journey to type safety is not about finding an absolute truth, but about diligently navigating its interpretations.

So, go forth, type your Python, and always question the 'truth' you're being told. Your code will thank you for it.