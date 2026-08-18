package tensorbank

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// CloneFile copies srcPath to dstPath. On Linux it tries a filesystem clone
// (FICLONE) so the extra GGUF does not consume a second full allocation until
// in-place reconstruct dirties pages; otherwise it falls back to a sequential
// copy. dstPath is replaced if it already exists.
func CloneFile(srcPath, dstPath string) error {
	if srcPath == "" || dstPath == "" {
		return fmt.Errorf("tensorbank: clone empty path")
	}
	absSrc, err1 := filepath.Abs(srcPath)
	absDst, err2 := filepath.Abs(dstPath)
	if err1 == nil && err2 == nil && absSrc == absDst {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dstPath), 0o755); err != nil {
		return err
	}
	if err := cloneFileFast(srcPath, dstPath); err == nil {
		return nil
	}
	return copyFileAll(srcPath, dstPath)
}

func copyFileAll(srcPath, dstPath string) error {
	in, err := os.Open(srcPath)
	if err != nil {
		return fmt.Errorf("tensorbank: clone open source: %w", err)
	}
	defer in.Close()
	st, err := in.Stat()
	if err != nil {
		return fmt.Errorf("tensorbank: clone stat source: %w", err)
	}
	out, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("tensorbank: clone create dest: %w", err)
	}
	defer func() {
		_ = out.Close()
	}()
	if err := copyFileRangeAll(int(in.Fd()), int(out.Fd()), 0, 0, st.Size()); err != nil {
		if _, seekErr := in.Seek(0, io.SeekStart); seekErr != nil {
			return fmt.Errorf("tensorbank: clone rewind: %w", seekErr)
		}
		if _, seekErr := out.Seek(0, io.SeekStart); seekErr != nil {
			return fmt.Errorf("tensorbank: clone dest rewind: %w", seekErr)
		}
		if err := out.Truncate(0); err != nil {
			return fmt.Errorf("tensorbank: clone dest truncate: %w", err)
		}
		if _, err := io.Copy(out, in); err != nil {
			return fmt.Errorf("tensorbank: clone copy: %w", err)
		}
	}
	if err := out.Sync(); err != nil {
		return fmt.Errorf("tensorbank: clone sync: %w", err)
	}
	return nil
}
