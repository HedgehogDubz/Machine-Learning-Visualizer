export function drawCircle(ctx, position, radius, opts = {}) {
    const { fillStyle, strokeStyle, lineWidth, fill = true, stroke = false, } = opts;
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
