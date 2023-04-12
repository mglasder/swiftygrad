import XCTest
@testable import Swiftygrad

final class EngineTests: XCTestCase {
    func testAdd() throws {
        
        let a = Value(data: 2.0) + Value(data: 3.0)
        let b = 2 + Value(data: 3.0)
        let c = Value(data: 2.0) + 3.0
        
        XCTAssertEqual(a.data, b.data)
        XCTAssertEqual(b.data, c.data)

    }
    
    func testSub() throws {
        
        let a = Value(data: 2.0) - Value(data: 3.0)
        let b = 2 - Value(data: 3.0)
        let c = Value(data: 2.0) - 3.0
        
        XCTAssertEqual(a.data, b.data)
        XCTAssertEqual(b.data, c.data)
    }
    
    func testMul() throws {
        
        let a = Value(data: 2.0) * Value(data: 3.0)
        let b = 2 * Value(data: 3.0)
        let c = Value(data: 2.0) * 3.0
        
        XCTAssertEqual(a.data, b.data)
        XCTAssertEqual(b.data, c.data)
    }
    
    func testDiv() throws {
        
        let a = Value(data: 2.0) / Value(data: 3.0)
        let b = 2 / Value(data: 3.0)
        let c = Value(data: 2.0) / 3.0
        
        XCTAssertEqual(a.data, b.data)
        XCTAssertEqual(b.data, c.data)
        
    }
    
    func testReLU() throws {
        
        let a = Value(data: -1.0)
        let b = a.relu()
        
        let c = Value(data: 2.0)
        let d = c.relu()
        
        XCTAssertEqual(b.data, 0)
        XCTAssertEqual(d.data, 2)
        
    }
    
    func testBackpopAdd() throws {
        
        let a = Value(data: 3.0)
        let b = Value(data: 1.0)
        
        let y = a + b
        
        y.backward()
        
        XCTAssertEqual(a.grad, 1.0)
        XCTAssertEqual(b.grad, 1.0)
        
    }
    
    func testBackpopLinear() throws {
        
        let a = Value(data: 3.0)
        let x = Value(data: 2.0)
        let b = Value(data: 1.0)
        
        let y = a * x + b
        
        y.backward()
        
        XCTAssertEqual(a.grad, x.data)
        XCTAssertEqual(x.grad, a.data)
        XCTAssertEqual(b.grad, 1.0)
    }
    
    func testBackpropPoly() throws {
        
        let a = Value(data: 3.0)
        let x = Value(data: 3.0)
        let b = Value(data: 1.0)
        
        let y = a * (x**2) - b
        
        y.backward()
        
        XCTAssertEqual(a.grad, pow(x.data, 2))
        XCTAssertEqual(x.grad, a.data * 2 * x.data)
        XCTAssertEqual(b.grad, -1.0)
        
    }
    
    
    func testBackpropDiv() throws {
        
        let a = Value(data: 3.0)
        let x = Value(data: 3.0)
        
        let y = a / x
        // y = a * x^-1   dy/dy = -a * x^-2
        
        y.backward()
        
        XCTAssertEqual(x.grad, -a.data / pow(x.data, 2))
        
    }
    
    func testBackpropReLU() throws {
       
        let a = Value(data: -1)
        let x = Value(data: 2.0)
        var y = a * x
        y = y.relu()
        y.backward()

        // y = ReLU(a * x) -> dy/dx = 0
        XCTAssertEqual(x.grad, 0)
        
        let a2 = Value(data: 1)
        let x2 = Value(data: 2.0)
        var y2 = a2 * x2
        y2 = y2.relu()
        y2.backward()

        XCTAssertEqual(x2.grad, 1)
        
    }
    
    func testLargeForwardBackwardPassExample() throws {
        
        let tol = 1e-6
        
        let a = Value(data: -4.0)
        let b = Value(data: 2.0)
        
        var c = a + b
        var d = (a * b) + (b**3)
        c = c + c + 1
        c = c + 1 + c - a
        d = d + (d * 2) + (b + a).relu()
        d = d + (3 * d) + (b - a).relu()
        let e = c - d
        let f = e**2
        var g = f / 2.0
        g = g + (10.0 / f)
        g.backward()
        
        XCTAssertEqual(a.data, -4.0, accuracy: tol)
        XCTAssertEqual(b.data, 2.0, accuracy: tol)
        XCTAssertEqual(g.data, 24.70408163265306, accuracy: tol)
    }
    
}
