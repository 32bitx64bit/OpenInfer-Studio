package convert

import (
	"fmt"
)

// vHeadPerm maps tiled (ggml) index → grouped (HF) index.
// HF stores V heads grouped by K head; ggml broadcasts tiled.
func vHeadPerm(nK, nVPerK, headDim int) []int {
	n := nK * nVPerK * headDim
	perm := make([]int, n)
	for k := 0; k < nK; k++ {
		for v := 0; v < nVPerK; v++ {
			for d := 0; d < headDim; d++ {
				src := k*(nVPerK*headDim) + v*headDim + d
				dst := v*(nK*headDim) + k*headDim + d
				perm[dst] = src
			}
		}
	}
	return perm
}

func applyReorder(data []byte, hfShape []int64, elem int, how string, h hyper) ([]byte, []int64, error) {
	if how == "" || h.linKeyHeads == 0 || h.linValHeads == 0 || h.linKeyHeads == h.linValHeads {
		return data, hfShape, nil
	}
	if h.linValHeads%h.linKeyHeads != 0 {
		return nil, nil, fmt.Errorf("linear_num_value_heads %d not divisible by linear_num_key_heads %d", h.linValHeads, h.linKeyHeads)
	}
	nVPerK := h.linValHeads / h.linKeyHeads
	switch how {
	case "v_qkv":
		if len(hfShape) != 2 {
			return nil, nil, fmt.Errorf("in_proj_qkv: expected rank-2, got %v", hfShape)
		}
		rows, cols := int(hfShape[0]), int(hfShape[1])
		qDim := h.linKeyDim * h.linKeyHeads
		kDim := qDim
		qk := qDim + kDim
		out, err := reorderQKVRows(data, rows, cols, elem, qk, h.linKeyHeads, nVPerK, h.linValDim)
		return out, hfShape, err
	case "v_rows":
		if len(hfShape) != 2 {
			return nil, nil, fmt.Errorf("in_proj_z: expected rank-2, got %v", hfShape)
		}
		out, err := permuteDim0(data, int(hfShape[0]), int(hfShape[1])*elem, vHeadPerm(h.linKeyHeads, nVPerK, h.linValDim))
		return out, hfShape, err
	case "v_rows1":
		if len(hfShape) != 2 {
			return nil, nil, fmt.Errorf("in_proj_a/b: expected rank-2, got %v", hfShape)
		}
		out, err := permuteDim0(data, int(hfShape[0]), int(hfShape[1])*elem, vHeadPerm(h.linKeyHeads, nVPerK, 1))
		return out, hfShape, err
	case "v_cols":
		if len(hfShape) != 2 {
			return nil, nil, fmt.Errorf("out_proj: expected rank-2, got %v", hfShape)
		}
		out, err := permuteDim1(data, int(hfShape[0]), int(hfShape[1]), elem, vHeadPerm(h.linKeyHeads, nVPerK, h.linValDim))
		return out, hfShape, err
	case "v_1d":
		n := len(data) / elem
		out, err := permuteDim0(data, n, elem, vHeadPerm(h.linKeyHeads, nVPerK, 1))
		return out, hfShape, err
	case "v_conv":
		if len(hfShape) != 2 {
			return nil, nil, fmt.Errorf("conv1d: expected rank-2 after squeeze, got %v", hfShape)
		}
		qkCh := h.linKeyDim * h.linKeyHeads * 2
		out, err := reorderQKVRows(data, int(hfShape[0]), int(hfShape[1]), elem, qkCh, h.linKeyHeads, nVPerK, h.linValDim)
		return out, hfShape, err
	default:
		return data, hfShape, nil
	}
}

func reorderQKVRows(data []byte, rows, cols, elem, qkRows, nK, nVPerK, headDim int) ([]byte, error) {
	rowBytes := cols * elem
	if qkRows < 0 || qkRows > rows {
		return nil, fmt.Errorf("qkv split qkRows=%d rows=%d", qkRows, rows)
	}
	head := data[:qkRows*rowBytes]
	v := data[qkRows*rowBytes:]
	vRows := rows - qkRows
	re, err := permuteDim0(v, vRows, rowBytes, vHeadPerm(nK, nVPerK, headDim))
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(data))
	copy(out, head)
	copy(out[len(head):], re)
	return out, nil
}

func permuteDim0(data []byte, n, stride int, perm []int) ([]byte, error) {
	if n != len(perm) {
		return nil, fmt.Errorf("reorder: dim0 n=%d perm=%d", n, len(perm))
	}
	if len(data) != n*stride {
		return nil, fmt.Errorf("reorder: payload %d want %d", len(data), n*stride)
	}
	out := make([]byte, len(data))
	for dst, src := range perm {
		copy(out[dst*stride:(dst+1)*stride], data[src*stride:(src+1)*stride])
	}
	return out, nil
}

func permuteDim1(data []byte, rows, cols, elem int, perm []int) ([]byte, error) {
	if cols != len(perm) {
		return nil, fmt.Errorf("reorder: dim1 cols=%d perm=%d", cols, len(perm))
	}
	if len(data) != rows*cols*elem {
		return nil, fmt.Errorf("reorder: payload %d want %d", len(data), rows*cols*elem)
	}
	out := make([]byte, len(data))
	for r := 0; r < rows; r++ {
		row := data[r*cols*elem : (r+1)*cols*elem]
		dst := out[r*cols*elem : (r+1)*cols*elem]
		for c, src := range perm {
			copy(dst[c*elem:(c+1)*elem], row[src*elem:(src+1)*elem])
		}
	}
	return out, nil
}

func squeezeConv(shape []int64, data []byte) ([]int64, []byte, error) {
	switch len(shape) {
	case 2:
		return append([]int64(nil), shape...), data, nil
	case 3:
		if shape[1] == 1 {
			return []int64{shape[0], shape[2]}, data, nil
		}
		if shape[2] == 1 {
			return []int64{shape[0], shape[1]}, data, nil
		}
	}
	return nil, nil, fmt.Errorf("conv1d: cannot squeeze shape %v", shape)
}
