import Foundation

public class Value: Hashable, Equatable {
    
    private var id = UUID()
    
    public var data: Double
    
    public var grad: Double = 0.0
    
    private var prev: Array<Value> = []
    private var op: String = ""
    private var _backward: () -> Void
    
    
    public init(data: Double, children: Array<Value> = [], op: String = "") {
        self.data = data
        self.prev = children
        self.op = op
        self._backward = {}
    }
    
    func backward() -> Void {
        
        var topo: [Value] = []
        var visited = Set<Value>()
        
        func buildTopo(_ v: Value) {
            if !visited.contains(v) {
                visited.insert(v)
                for child in v.prev {
                    buildTopo(child)
                }
                topo.append(v)
            }
        }
        
        buildTopo(self)
        
        self.grad = 1
        for v in topo.reversed() {
            v._backward()
        }
    }
    
}


extension Value {

    // MARK: Addition
    static func + (lhs: Value, rhs: Value) -> Value {
                
        let out = Value(data: lhs.data + rhs.data, children: [lhs, rhs], op: "+")

        func _backward() {
            lhs.grad += out.grad
            rhs.grad += out.grad
        }

        out._backward = _backward

        return out
    }
    
    static func + (lhs: Value, rhs: Double) -> Value {
        return lhs + Value(data: rhs)
    }
    
    static func + (lhs: Double, rhs: Value) -> Value {
        return Value(data: lhs) + rhs
    }
    
    
    // MARK: Subtraction
    static func - (lhs: Value, rhs: Value) -> Value {
        return lhs + (rhs * -1)
    }
    
    static func - (lhs: Value, rhs: Double) -> Value {
        return lhs - Value(data: rhs)
    }
    
    static func - (lhs: Double, rhs: Value) -> Value {
        return Value(data: lhs) - rhs
    }
    
    
    // MARK: Multiplication
    static func * (lhs: Value, rhs: Value) -> Value {
        let out = Value(data: lhs.data * rhs.data, children: [lhs, rhs], op: "*")
        
        func _backward() {
            lhs.grad = rhs.data * out.grad
            rhs.grad = lhs.data * out.grad
        }

        out._backward = _backward
        
        return out
    }
    
    static func * (lhs: Value, rhs: Double) -> Value {
        return lhs * Value(data: rhs)
    }
    
    static func * (lhs: Double, rhs: Value) -> Value {
        return Value(data: lhs) * rhs
    }
    
    
    
    // MARK: Power
    static func ** (lhs: Value, rhs: Double) -> Value {
        let out = Value(data: pow(lhs.data, rhs) , children: [lhs,], op: "**\(rhs)")
        
        func _backward() {
            lhs.grad += (rhs * pow(lhs.data,(rhs-1))) * out.grad
        }

        out._backward = _backward
        
        return out
    }

    
    // MARK: Division
    static func / (lhs: Value, rhs: Value) -> Value {
        return lhs * (rhs ** -1)
    }
    
    static func / (lhs: Value, rhs: Double) -> Value {
        return lhs / Value(data: rhs)
    }
    
    static func / (lhs: Double, rhs: Value) -> Value {
        return Value(data: lhs) / rhs
    }
    
    // MARK: Relu
    func relu() -> Value {
        let out = Value(data: self.data < 0 ? 0 : self.data, children: [self], op: "ReLU")
        
        func _backward() {
            self.grad += (out.data > 0 ? 1: 0) * out.grad
        }

        out._backward = _backward
        
        return out
    }

}


extension Value {
    
    // MARK: Conform to Hashable and Equatable
    /// TODO:  replace with hash(into: ...)
    public var hashValue: Int {
        return id.hashValue
    }

    public static func == (lhs: Value, rhs: Value) -> Bool {
        return lhs.hashValue == rhs.hashValue
    }
    
    // MARK: Description
    var description: String {
        "Value(data: \(data), grad: \(grad))"
    }
    
}


infix operator **
