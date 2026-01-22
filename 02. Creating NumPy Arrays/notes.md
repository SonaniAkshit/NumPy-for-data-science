# 2: Creating NumPy Arrays

## Big Picture First (don’t skip)

Creating arrays is not about “getting numbers into NumPy”.

It is about:

* Controlling **shape**
* Controlling **initial values**
* Controlling **memory and speed**
* Preparing data for **ML models, testing, and math**

Different creation functions exist because **different problems need different guarantees**.

If you randomly choose functions, you will:

* Introduce silent bugs
* Break ML models
* Fail shape-based interview questions

---

## 1. `np.array()` – converting existing data

### WHY it exists

`np.array()` exists to **convert Python data structures** into NumPy arrays.

Lists, tuples, nested lists, ranges, all come here first.

### Basic example

```python
import numpy as np

arr = np.array([1, 2, 3, 4])
```

* Shape: `(4,)` → 1D array
* dtype: inferred automatically

### 2D example

```python
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])
```

* Shape: `(2, 3)`
* 2 rows, 3 columns

![Image](https://i.sstatic.net/NWTQH.png)

![Image](https://iq.opengenus.org/content/images/2020/04/index.png)

### Real data science use

* Converting CSV rows (after reading)
* Turning feature lists into arrays
* Initial experiments and debugging

### Common beginner mistakes

❌ Assuming dtype will always be int
❌ Passing uneven nested lists
❌ Ignoring shape after creation

```python
np.array([1, 2, 3.5])  # becomes float
```

This **matters** in ML.

### Interview pitfall

> “`np.array()` copies data or creates a view?”

Answer:
**It usually creates a copy**, unless explicitly told otherwise.

---

## 2. `np.zeros()` – controlled empty baseline

### WHY it exists

To create arrays **with known shape and guaranteed zero values**.

This is critical for:

* ML weights
* Accumulators
* Placeholders

### Example

```python
np.zeros(5)
```

Shape: `(5,)`

```python
np.zeros((3, 4))
```

Shape: `(3, 4)`

![Image](https://numpy.org/devdocs/_images/np_ones_zeros_matrix.png)

![Image](https://numpy.org/devdocs/_images/np_matrix_indexing.png)

### Real ML use

```python
weights = np.zeros((100, 10))
```

You want:

* Correct shape
* Clean starting point
* No garbage values

### Common mistake

❌ Using Python lists for initialization
❌ Forgetting tuple for multi-dimensional shape

```python
np.zeros(3, 4)  # ❌ wrong
```

### Interview note

Zeros are **deterministic**. That matters for debugging and reproducibility.

---

## 3. `np.ones()` – same logic, different intent

### WHY it exists

Same reason as zeros, but with **ones**.

Often used when:

* You want bias terms
* You want to test formulas
* You want to avoid zero-multiplication issues

### Example

```python
np.ones((2, 3))
```

![Image](https://numpy.org/devdocs/_images/np_ones_zeros_random.png)

![Image](https://www.w3resource.com/w3r_images/numpy-array-ones-function-image-1.png)

### ML example

```python
bias = np.ones((1, n_features))
```

### Beginner mistake

❌ Thinking ones are just “demo data”

They are **intentional initial values**.

---

## 4. `np.empty()` – dangerous but fast

### WHY it exists

To allocate memory **without initializing values**.

This is about **performance**, not convenience.

### Example

```python
np.empty((2, 3))
```

![Image](https://cdn.educba.com/academy/wp-content/uploads/2020/08/NumPy-Empty-output.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ANz0HQW-oKVrzq5gD4b2geQ.png)

You will see **random values**.

### When to use

* High-performance pipelines
* When you immediately overwrite all values

### When NOT to use

* Beginners
* ML initialization
* Any code where values matter before assignment

### Interview trap

> “Is `np.empty()` faster than `np.zeros()`?”

Yes.
But using it blindly is a **bug factory**.

---

## 5. `np.arange()` – range with step (but dangerous)

### WHY it exists

To generate sequences with a **fixed step size**.

### Example

```python
np.arange(0, 10, 2)
```

Output:

```python
[0 2 4 6 8]
```

![Image](https://datascienceparichay.com/wp-content/uploads/2021/08/numpy-arange-illustration.png)

![Image](https://www.w3resource.com/w3r_images/numpy-arange-function-image-1.png)

### Common real use

* Index generation
* Loop replacement
* Discrete steps

### Major problem (important)

```python
np.arange(0, 1, 0.1)
```

You **cannot trust floating precision** here.

### Interview warning

If you use `arange` with floats in production ML code, you are asking for bugs.

---

## 6. `np.linspace()` – precision-safe ranges

### WHY it exists

To generate **exactly N evenly spaced values** between two numbers.

This avoids floating precision problems.

### Example

```python
np.linspace(0, 1, 5)
```

Output:

```python
[0.   0.25 0.5  0.75 1.  ]
```

![Image](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/02/numpy-linspace-illustration.png)

![Image](https://www.kodeclik.com/assets/numpy-linspace-grid-of-points.webp)

### Real use

* Plotting
* Mathematical simulations
* ML feature grids

### Interview comparison

* `arange` → step based
* `linspace` → count based

If you confuse these, interviewers notice immediately.

---

## 7. `np.eye()` – identity matrix (linear algebra core)

### WHY it exists

To create **identity matrices** used in:

* Linear algebra
* ML regularization
* Matrix math

### Example

```python
np.eye(3)
```

![Image](https://www.kodeclik.com/assets/numpy-eye-example.webp)

![Image](https://datascienceparichay.com/wp-content/uploads/2022/12/identity-matrix-example-1024x538.png)

Output:

```
1 0 0
0 1 0
0 0 1
```

### ML relevance

* Ridge regression
* Matrix inversion stability
* Feature decorrelation

### Beginner mistake

❌ Thinking this is rare or academic
It’s not. It shows up in real ML math.

---

## Shape discipline (non-negotiable)

Every creation function forces you to think in **shapes**.

![Image](https://towardsdatascience.com/wp-content/uploads/2024/09/1T8BqVvPTcyuXbzWubPt8Zw.png)

![Image](https://i0.wp.com/indianaiproduction.com/wp-content/uploads/2019/06/NumPy-shape.png?resize=1200%2C488\&ssl=1)

If you cannot instantly say:

* how many rows
* how many columns
* total elements

You are not ready for Pandas or ML.

---

## Practice Section (MANDATORY – 20)

### Easy (1–7)

1. Why does NumPy provide multiple array creation functions?
2. Shape of `np.zeros((4, 5))`?
3. Difference between `np.array()` and `np.zeros()`?
4. Output shape of `np.ones(10)`?
5. Why is `np.empty()` risky?
6. Which function guarantees evenly spaced floats?
7. What does `np.eye(4)` create?

### Medium (8–14)

8. Predict shape:

   ```python
   np.array([[1,2],[3,4],[5,6]])
   ```
9. Why is `np.arange()` unsafe with floats?
10. When would you prefer `linspace` over `arange`?
11. What happens if you forget tuple in `np.zeros(3,4)`?
12. ML use case for `np.ones()`?
13. Why does `np.array()` infer dtype?
14. Which function is fastest to allocate memory?

### Hard (15–20)

15. Why is array initialization important in ML?
16. What silent bugs can `np.empty()` cause?
17. Why identity matrices matter in regression?
18. How can wrong shape break model training?
19. Why interviewers ask about `linspace` vs `arange`?
20. Design a test array for ML weights and explain why.

---

## Industry & Real Data Science Tasks

![Image](https://pyimagesearch.com/wp-content/uploads/2021/04/weight-initialization-for-neural-networks.png)

![Image](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-020-2649-2/MediaObjects/41586_2020_2649_Fig1_HTML.png)

![Image](https://towardsdatascience.com/wp-content/uploads/2024/09/1T8BqVvPTcyuXbzWubPt8Zw.png)

1. Initialize ML weights
2. Create dummy datasets
3. Test preprocessing logic
4. Create feature grids
5. Build identity matrices for regularization
6. Debug shape mismatches
7. Benchmark performance
8. Simulate numerical experiments

---

## Final hard truth

If you randomly use:

* `array`
* `zeros`
* `empty`
* `arange`

You are **guessing**, not engineering.

From now on, every time you create an array, ask yourself:

> Why THIS function, not the others?

If you can answer that, you’re learning correctly.