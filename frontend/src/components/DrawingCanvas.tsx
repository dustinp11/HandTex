import React, { useMemo, useRef, useState } from "react";
import { View, PanResponder } from "react-native";
import Svg, { Polyline } from "react-native-svg";

export type Point = { x: number; y: number };
export type Stroke = Point[];

export default function DrawingCanvas({
  strokes,
  onChangeStrokes,
  strokeWidth = 4,
}: {
  strokes: Stroke[];
  onChangeStrokes: (next: Stroke[]) => void;
  strokeWidth?: number;
}) {
  const [activeStroke, setActiveStroke] = useState<Stroke>([]);
  const activeStrokeRef = useRef<Stroke>([]); // ✅ always current

  const panResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,

        onPanResponderGrant: (evt) => {
          const { locationX, locationY } = evt.nativeEvent;
          const start: Stroke = [{ x: locationX, y: locationY }];
          activeStrokeRef.current = start;
          setActiveStroke(start);
        },

        onPanResponderMove: (evt) => {
          const { locationX, locationY } = evt.nativeEvent;
          const next = [...activeStrokeRef.current, { x: locationX, y: locationY }];
          activeStrokeRef.current = next;
          setActiveStroke(next);
        },

        onPanResponderRelease: () => {
          const stroke = activeStrokeRef.current;
          if (stroke.length > 1) {
            onChangeStrokes([...strokes, stroke]);
          }
          activeStrokeRef.current = [];
          setActiveStroke([]);
        },

        onPanResponderTerminate: () => {
          // if gesture gets interrupted
          activeStrokeRef.current = [];
          setActiveStroke([]);
        },
      }),
    [strokes, onChangeStrokes] // ✅ no activeStroke dependency
  );

  return (
    <View style={{ flex: 1, backgroundColor: "white" }} {...panResponder.panHandlers}>
      <Svg width="100%" height="100%">
        {strokes.map((stroke, idx) => (
          <Polyline
            key={idx}
            points={stroke.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke="black"
            strokeWidth={strokeWidth}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}

        {activeStroke.length > 0 && (
          <Polyline
            points={activeStroke.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke="black"
            strokeWidth={strokeWidth}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}
      </Svg>
    </View>
  );
}
