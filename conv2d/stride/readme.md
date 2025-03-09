# Understanding Convolutional Kernel Movement with Stride

## **1. Given Matrices**

### **Input Matrix (5x5)**
```
1   2   3   4   5
6   7   8   9  10
11  12  13  14  15
16  17  18  19  20
21  22  23  24  25
```

### **Kernel (3x3)**
```
1   0  -1
1   0  -1
1   0  -1
```

### **Stride = 2**
- The kernel moves **2 pixels** at a time instead of 1.
- No padding is applied.

---

## **2. Step-by-Step Kernel Movement**

### **Formula for convolution at each step:**
```
Sum of (Element-wise multiplication between Kernel & Region)
```

### **Step 1: Apply Kernel at (0,0)**
#### **Extract 3x3 region:**
```
1   2   3
6   7   8
11  12  13
```
#### **Apply Convolution:**
```
(1*1) + (2*0) + (3*-1) +
(6*1) + (7*0) + (8*-1) +
(11*1) + (12*0) + (13*-1)
= -6
```

---

### **Step 2: Move Right (stride=2) → Apply Kernel at (0,2)**
#### **Extract 3x3 region:**
```
3   4   5
8   9   10
13  14  15
```
#### **Apply Convolution:**
```
(3*1) + (4*0) + (5*-1) +
(8*1) + (9*0) + (10*-1) +
(13*1) + (14*0) + (15*-1)
= -6
```

---

### **Step 3: Move Down (stride=2) → Apply Kernel at (2,0)**
#### **Extract 3x3 region:**
```
11  12  13
16  17  18
21  22  23
```
#### **Apply Convolution:**
```
(11*1) + (12*0) + (13*-1) +
(16*1) + (17*0) + (18*-1) +
(21*1) + (22*0) + (23*-1)
= -6
```

---

### **Step 4: Move Right (stride=2) → Apply Kernel at (2,2)**
#### **Extract 3x3 region:**
```
13  14  15
18  19  20
23  24  25
```
#### **Apply Convolution:**
```
(13*1) + (14*0) + (15*-1) +
(18*1) + (19*0) + (20*-1) +
(23*1) + (24*0) + (25*-1)
= -6
```

---

## **3. Final Output Matrix (2x2)**
```
-6  -6
-6  -6
```
This matches the expected result! ✅

---

## **4. Key Takeaways**
1. **Stride controls how much the kernel moves.**
   - If **stride=1**, the kernel moves **pixel by pixel**.
   - If **stride=2**, the kernel moves **2 pixels at a time**, making the output **smaller**.
   
2. **Kernel applies element-wise multiplication** over input patches and sums the results.

3. **Larger stride means fewer steps, reducing output size.**
   - **Stride=1** → `(3x3)` output.
   - **Stride=2** → `(2x2)` output.

---

## **5. Next Steps**
🔹 Try **stride=1** and compare the result.
🔹 Add **padding=1** and check how the output size changes.
🔹 Use **different kernels** and experiment!



