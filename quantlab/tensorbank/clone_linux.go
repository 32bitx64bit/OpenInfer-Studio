//go:build linux

package tensorbank

import (
	"os"

	"golang.org/x/sys/unix"
)

func cloneFileFast(srcPath, dstPath string) error {
	in, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer out.Close()
	if err := unix.IoctlSetInt(int(out.Fd()), unix.FICLONE, int(in.Fd())); err != nil {
		return err
	}
	return out.Sync()
}
