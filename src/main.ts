import * as gui from "./gui/gui.ts";
import * as ml from "./mathlib/math.ts";

const enum UiAction {
	placeholder,
	increment,
	decrement,
	drag_seed
}

//const a = new ml.Vector(new Float32Array([1, 2, 3, 4, 5]));
//const b = new ml.Vector(new Float32Array([1, 2, 3, 4, 5]));
//
//console.log(ml.Vector.add(a, b));
//console.log("dot", ml.Vector.dot(a, b));

const vec = new ml.Vector(new Float32Array([1, 2, 3]))
const m = new ml.Matrix([
	new ml.Vector(new Float32Array([1, 1, 4])),
	new ml.Vector(new Float32Array([2, 2, 3])),
	new ml.Vector(new Float32Array([3, 1, 3]))
]);

console.log("mult", ml.Matrix.multVector(m, vec));
console.log("trans", ml.Matrix.multTransposeVector(m, vec));

function test_rng(s: number) {
	const seed = s;
	const rng = new ml.PCG32(BigInt(seed));
	console.log(rng.pcg32_0to1());
}

const c = <gui.REND>gui.canvas.getContext("2d");
const input_state = new gui.InputState(gui.canvas, 0, 0)

let num = 0;
let seed = 1;

const update = () => {

	gui.updateCanvasSizing();
	const stack = new gui.Stack<gui.N<UiAction>>();

	const w = stack.makeWindow(c, input_state, { window: UiAction.placeholder, header: UiAction.placeholder, resizeable: UiAction.placeholder, close_btn: null }, { title: "neural test", width: 300, height: 200 });

	w.makeLabel(c, null, "num = " + num);
	w.makeButton(c, UiAction.increment, "increment");
	w.makeButton(c, UiAction.decrement, "decrement");

	w.makeLabel(c, null, "seed = " + seed);
	w.makeDraggable(c, UiAction.drag_seed, "drag seed");

	const ret = stack.requestAction(input_state);
	const action = ret.action;


	switch (action) {
		case UiAction.increment:
			num++;
			break;
		case UiAction.decrement:
			num--;
			break;
		case UiAction.drag_seed:
			seed = gui.updateDraggableValue(seed, input_state, 1.0);
			break;
	}

	if (action != null)
		test_rng(seed);

	c.clearRect(0, 0, c.canvas.width, c.canvas.height);
	stack.stack_render(c, input_state);

	input_state.end();

	requestAnimationFrame(update);
}

update();
