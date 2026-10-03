//go:build linux

package processes

import "testing"

func TestCoreDumpingStatus(t *testing.T) {
	for _, tc := range []struct {
		status string
		want   bool
	}{
		{"Name:\tsd-server\nState:\tD (disk sleep)\nCoreDumping:\t1\nThreads:\t24\n", true},
		{"Name:\tsd-server\nState:\tS (sleeping)\nCoreDumping:\t0\n", false},
		{"State:\tD (disk sleep)\n", false},
		{"CoreDumping:\t10\n", false},
	} {
		if got := coreDumpingStatus(tc.status); got != tc.want {
			t.Fatalf("status %q: got %v, want %v", tc.status, got, tc.want)
		}
	}
	var h *Handle
	if h.IsCoreDumping() {
		t.Fatal("missing process reported a crash")
	}
}
