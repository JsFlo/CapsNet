package fhc.tfsandbox.capsnettweak.DrawImageView

import android.graphics.Canvas
import android.graphics.Paint

data class CanvasModel(val width: Int, val height: Int, val lines: MutableList<CanvasLine> = mutableListOf<CanvasLine>()) {
    // mutable but happening on draw so
    var currentLine: CanvasLine? = null

    fun startLine(x: Float, y: Float) {
        currentLine = CanvasLine(x, y)
        lines.add(currentLine!!)
    }

    fun endLine() {
        currentLine = null
    }

    fun addLineSegmentToCurrentLine(x: Float, y: Float) {
        currentLine?.add(x, y)
    }

    fun draw(canvas: Canvas, paint: Paint, startLineIndex: Int) {
        lines.filterIndexed { i, _ -> i >= startLineIndex }
                .forEach {
                    it.draw(canvas, paint)
                }
    }
}

data class CanvasLine(val canvasLineSegments: MutableList<CanvasLineSegment>) {
    constructor(x: Float, y: Float) : this(mutableListOf(CanvasLineSegment(x, y)))

    fun add(x: Float, y: Float) {
        canvasLineSegments.add(CanvasLineSegment(x, y))
    }

    fun draw(canvas: Canvas, paint: Paint) {
        val segmentSize = canvasLineSegments.size
        if (segmentSize > 1) {
            // mutable for on draw, not sure if needed
            var tempPair: MutablePair = MutablePair(0f, 0f)
            // an excuse to use fold right but should probably change it
            canvasLineSegments.foldRightIndexed(tempPair, { index, canvasLineSegment, acc ->
                if (acc.x != 0f && acc.y != 0f) {
                    canvas.drawLine(acc.x, acc.y, canvasLineSegment.x, canvasLineSegment.y, paint)
                }
                tempPair.x = canvasLineSegment.x
                tempPair.y = canvasLineSegment.y
                tempPair
            })
        }
    }
}

data class CanvasLineSegment(val x: Float, val y: Float)

class MutablePair(var x: Float, var y: Float)