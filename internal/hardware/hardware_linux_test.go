//go:build linux

package hardware

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSysfsVRAMTotal(t *testing.T) {
	dir := t.TempDir()
	if got := sysfsVRAMTotal(dir); got != 0 {
		t.Fatalf("missing mem_info_vram_total: got %d, want 0", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "mem_info_vram_total"), []byte("17163091968\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := sysfsVRAMTotal(dir); got != 17163091968 {
		t.Fatalf("got %d, want 17163091968", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "mem_info_vram_total"), []byte("garbage"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := sysfsVRAMTotal(dir); got != 0 {
		t.Fatalf("unparseable: got %d, want 0", got)
	}
}
