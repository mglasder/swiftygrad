# Swiftygrad

Swiftygrad is a simple autograd library for Swift inspired by Andrej Kaparthy's micrograd for Python (see https://github.com/karpathy/micrograd).


### Basic usage

```

let a = Value(data: 3.0)
let x = Value(data: 2.0)
let b = Value(data: 1.0)

let y = a * x + b

y.backward()

// a.grad == 2.0
// x.grad == 3.0
// b.grad == 1.0

```


### Future features

- Add functionality to define a basic MLP
- Add gradient decent to optimize MPLs


