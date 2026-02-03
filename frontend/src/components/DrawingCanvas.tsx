import React, { useMemo, useState } from "react";
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

  const panResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: () => true,

        onPanResponderGrant: (evt) => {
          const { locationX, locationY } = evt.nativeEvent;
          setActiveStroke([{ x: locationX, y: locationY }]);
        },

        onPanResponderMove: (evt) => {
          const { locationX, locationY } = evt.nativeEvent;
          setActiveStroke((prev) => [...prev, { x: locationX, y: locationY }]);
        },

        onPanResponderRelease: () => {
          if (activeStroke.length > 1) onChangeStrokes([...strokes, activeStroke]);
          setActiveStroke([]);
        },
      }),
    [strokes, activeStroke, onChangeStrokes]
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
