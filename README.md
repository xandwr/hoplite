# Hoplite

> Code for the screen, not the platform, with
> a single, unyielding dependency.

```rs
use hoplite::*;

fn main() {
    run(|canvas| {
        canvas.background(BLACK);
        canvas.circle(100, 100, 50);
    });
}
```

> One closure, one call.

## The escape hatch...

```rs
run_with(Config { width: 800, height: 600, title: "My Game" }, |canvas| {
    // same API
});
```

## My "sane" defaults

1. Coordinates are just numbers.

```rs
canvas.rect(10, 20, 100, 50); // no more Rect::new(), no Point { x, y }, no bullshit
```

2. Colors are words first, values second

```rs
canvas.fill(RED);
canvas.fill(rgb(255, 128, 0));
canvas.fill("#ff8800"); // why not
```

3. State is implicit (like Processing)

```rs
canvas.fill(BLUE);
canvas.stroke(WHITE);
canvas.stroke_weight(3);
canvas.rect(10, 10, 100, 100); // uses current fill/stroke
```

4. Animation is just... returning

```rs
fn main() {
    let mut x = 0.0;
    
    run(|canvas| {
        canvas.background(BLACK);
        canvas.circle(x, 100, 20);
        x += 1.0;  // state lives in the closure's environment
    });
}
```

5. Input is really freakin' simple

```rs
run(|canvas| {
    if canvas.key_down(Key::Space) {
        // jump
    }
    if canvas.mouse_pressed() {
        canvas.circle(canvas.mouse_x(), canvas.mouse_y(), 10);
    }
});
```