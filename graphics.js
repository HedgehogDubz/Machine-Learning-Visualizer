"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.drawCircle = drawCircle;
function drawCircle(ctx, position, radius, opts) {
    if (opts === void 0) { opts = {}; }
    var fillStyle = opts.fillStyle, strokeStyle = opts.strokeStyle, lineWidth = opts.lineWidth, _a = opts.fill, fill = _a === void 0 ? true : _a, _b = opts.stroke, stroke = _b === void 0 ? false : _b;
    ctx.beginPath();
    ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
    if (lineWidth !== undefined)
        ctx.lineWidth = lineWidth;
    if (fillStyle !== undefined)
        ctx.fillStyle = fillStyle;
    if (strokeStyle !== undefined)
        ctx.strokeStyle = strokeStyle;
    if (fill)
        ctx.fill();
    if (stroke)
        ctx.stroke();
}
