# https://www.kernel.org/doc/Documentation/kbuild/makefiles.txt

KERNEL := $(shell uname -r)
LINUX_DIR := /lib/modules/$(KERNEL)/build

# Select driver with e.g.:
#   make install DRIVER=cloud
#   make install DRIVER=rpm
DRIVER ?= current

ifeq ($(DRIVER),current)
DRIVER_SOURCE := imx385.c
else ifeq ($(DRIVER),rpm)
DRIVER_SOURCE := RPM_cloud_imx385_driver/RPM_imx385.c
else ifeq ($(DRIVER),cloud)
DRIVER_SOURCE := RPM_cloud_imx385_driver/cloud_imx385.c
else
DRIVER_SOURCE := $(DRIVER)
endif

BUILD_DIR := .build/$(basename $(notdir $(DRIVER_SOURCE)))
KBUILD_FILE := $(BUILD_DIR)/Makefile

.PHONY: all modules install overlay clean

all: install overlay

$(KBUILD_FILE): $(DRIVER_SOURCE)
	mkdir -p $(BUILD_DIR)
	cp $(DRIVER_SOURCE) $(BUILD_DIR)/imx385.c
	printf 'obj-m += imx385.o\n' > $(KBUILD_FILE)

modules: $(KBUILD_FILE)
	$(MAKE) -C $(LINUX_DIR) M='$(PWD)/$(BUILD_DIR)' modules

install: modules
	sudo mkdir -p /lib/modules/$(KERNEL)/extra
	sudo cp $(BUILD_DIR)/imx385.ko /lib/modules/$(KERNEL)/extra/
	sudo depmod -a

overlay:
	dtc -I dts -O dtb -@ -o imx385-overlay.dtbo imx385-overlay.dts
	sudo cp imx385-overlay.dtbo /boot/firmware/overlays/

clean:
	$(MAKE) -C $(LINUX_DIR) M='$(PWD)/$(BUILD_DIR)' clean