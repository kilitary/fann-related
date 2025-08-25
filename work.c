// what u gonna do when wh off your line, what u go nn ad o?
#include <windows.h>
#include <mmsystem.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>

int main(void)
{
    int ret;
    int file_count = 0;
    int max_iterations = 1000; // Prevent infinite loop
    int current_iteration = 0;

		srand(time(NULL));
    HANDLE fh;
    WIN32_FIND_DATA fd;
		int c=0;
		int curc=0;
    
    while (current_iteration < max_iterations)
    {
        current_iteration++;
        
        // First pass: count files
        file_count = 0;
        memset(&fd, 0x0, sizeof(fd));
        fh = FindFirstFile("e:\\work\\*.wav", &fd);
        if (fh != INVALID_HANDLE_VALUE)
        {
            do
            {
                file_count++;
            }
            while (FindNextFile(fh, &fd));
            FindClose(fh);
        }
        
        if (file_count == 0)
        {
            printf("No .wav files found in e:\\work\\\n");
            break;
        }
        
        // Select random file index
			c = rand() % file_count;
			
        // Second pass: get the selected file
        memset(&fd, 0x0, sizeof(fd));
        fh = FindFirstFile("e:\\work\\*.wav", &fd);
        // deb("dump fh: %x", fh);
				curc=c;
        if (fh != INVALID_HANDLE_VALUE)
        {
            // Skip to the selected file
			      while (curc > 0 && FindNextFile(fh, &fd))
            {
                curc--;
            }
						printf("Playing file %u: %s\n", c, fd.cFileName);
            
            char fn[MAX_PATH];
				    sprintf(fn,"e:\\work\\%s",fd.cFileName);
            ret=PlaySound(fn,NULL,SND_FILENAME);
            if (!ret)
            {
                printf("Failed to play sound: %s (Error: %d)\n", fn, GetLastError());
            }
            
            FindClose(fh);
        }
				curc=0;
       // printf("%d %d",ret,GetLastError());
       
       // Add a small delay to prevent excessive CPU usage
       Sleep(1000);
    }
    
    printf("Program completed after %d iterations\n", current_iteration);
    return 0;
}

// just because not higher sh e
// 2025: still fighting 
